import time
import os
import argparse
import random
import numpy as np
import scipy.io
import torch as T

import Classes.Environment_Platoon as ENV
from Classes.buffer import ReplayBuffer
from agent import IndependentDQNAgent

# ===================================================================== #
# Independent Double-DQN (task-decomposed Q-heads) on the Parvini AoI-MARL
# platoon C-V2X environment. CLEAN DISCRETE BASELINE:
#   - environment = verbatim Parvini (soft AoI penalty -AoI/20 in task-2);
#   - learner     = fresh per-platoon Independent Double-DQN (no global critic,
#                   no CMDP). Action space is enumerated discretely:
#                   n_actions = n_RB * n_S(=2) * n_power.
# This is steps 1 (DDPG->DQN), 2a (renewal cadence flag) and a pre-emptive
# step-4 (smaller net). Steps 3 (scale) and 5 (per-platoon Lagrangian on the
# SAME argmax) build on top of this. See README.md.
# Example:
#   python Main.py --episodes 600 --seed 2 --renew_every 20 --out_tag re20   # validate vs DDPG-soft
#   python Main.py --episodes 600 --seed 2 --renew_every 1  --out_tag re1    # step 2a
#   python Main.py --smoke                                                    # wiring test (seconds)
# ===================================================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=500)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--tau', type=float, default=8.0,
                    help='AoI threshold for the LOGGED violation rate (soft mode: NOT used in the reward)')
parser.add_argument('--renew_every', type=int, default=20,
                    help='renew positions/slow-fading every N episodes (Parvini=20; per-episode=1)')
parser.add_argument('--n_RB', type=int, default=3)
parser.add_argument('--n_veh', type=int, default=20, help='platoons = n_veh / 4 (max 8 platoons, env position cap)')
parser.add_argument('--n_power', type=int, default=30, help='discrete power levels spanning 1..max_power dBm')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--buffer', type=int, default=50000)
parser.add_argument('--fc1', type=int, default=256)
parser.add_argument('--fc2', type=int, default=128)
parser.add_argument('--target_tau', type=float, default=0.005, help='Polyak soft-update rate for target nets')
parser.add_argument('--eps_start', type=float, default=1.0)
parser.add_argument('--eps_end', type=float, default=0.05)
parser.add_argument('--eps_decay_frac', type=float, default=0.5,
                    help='fraction of total env steps over which epsilon decays linearly to eps_end')
parser.add_argument('--out_tag', type=str, default='')
parser.add_argument('--out_subdir', type=str, default='',
                    help='optional subfolder under model/ for all outputs (e.g. Scan_Cadence)')
# [RQ1-CMDP] step-5: per-platoon Lagrangian constraint. soft = AoI as -AoI/20 reward penalty
# (validated baseline); hard = AoI as a per-platoon CMDP constraint P(AoI>tau)<=eps via a cost
# Q-head + per-platoon lambda_j, greedy = argmax_a [Q1 + Q2 - lambda_j * Q^c].
parser.add_argument('--mode', choices=['soft', 'hard'], default='soft')
parser.add_argument('--eps', type=float, default=0.10, help='[RQ1-CMDP] target violation probability epsilon')
parser.add_argument('--eta_lam', type=float, default=1.0, help='[RQ1-CMDP] integral dual step size')
parser.add_argument('--lam_max', type=float, default=20.0, help='[RQ1-CMDP] multiplier clip')
parser.add_argument('--dual', choices=['integral', 'pid'], default='pid', help='[RQ1-CMDP] dual update rule')
parser.add_argument('--kp', type=float, default=1.0, help='[RQ1-CMDP] PID proportional gain')
parser.add_argument('--ki', type=float, default=1.0, help='[RQ1-CMDP] PID integral gain')
parser.add_argument('--kd', type=float, default=0.5, help='[RQ1-CMDP] PID derivative gain')
parser.add_argument('--smoke', action='store_true', help='tiny end-to-end wiring test (NOT a result)')
args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
T.manual_seed(SEED)
T.cuda.manual_seed_all(SEED)

# ---- scenario / lanes (verbatim Parvini, urban Annex A of 3GPP TS 36.885) ----
up_lanes = [i / 2.0 for i in [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i / 2.0 for i in [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2, 750 - 3.5 / 2]]
left_lanes = [i / 2.0 for i in [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i / 2.0 for i in [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2, 1299 - 3.5 / 2]]
width = 750 / 2
height = 1298 / 2

size_platoon = 4
n_veh = args.n_veh
n_platoon = int(n_veh / size_platoon)
n_RB = args.n_RB
n_S = 2
Gap = 25
max_power = 30
V2I_min = 540
bandwidth = int(180000)
V2V_size = int(4000 * 8)

n_power = args.n_power
n_actions = n_RB * n_S * n_power            # discrete action cardinality per agent (default 3*2*30=180)


def decode_action(idx):
    """action index -> (RB, mode, power_dBm). mode 0=inter/V2I, 1=intra/V2V."""
    rb = idx // (n_S * n_power)
    rem = idx % (n_S * n_power)
    mode = rem // n_power
    p_level = rem % n_power
    if n_power == 1:
        power = max_power
    else:
        power = 1 + round(p_level * (max_power - 1) / (n_power - 1))
    return int(rb), int(mode), int(power)


env = ENV.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height,
                  n_veh, size_platoon, n_RB, V2I_min, bandwidth, V2V_size, Gap)
env.new_random_game()
# [RQ1-CMDP] AoI handling: soft keeps the -AoI/20 reward penalty; hard turns it OFF (AoI -> constraint).
env.aoi_penalty_coef = (1.0 / 20) if args.mode == 'soft' else 0.0

n_episode = args.episodes
n_step_per_episode = int(env.time_slow / env.time_fast)   # = 100 (physics-locked: 100ms CAM frame / 1ms slot).
n_episode_test = 100                                       # rolling window for the per-step .mat logs
if args.smoke:
    n_episode = 4
    n_step_per_episode = 20


def get_state(env, idx):
    """ Get state for platoon idx (verbatim Parvini state construction). """
    V2I_abs = (env.V2I_channels_abs[idx * size_platoon] - 60) / 60.0
    V2V_abs = (env.V2V_channels_abs[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1))] - 60) / 60.0
    V2I_fast = (env.V2I_channels_with_fastfading[idx * size_platoon, :] - env.V2I_channels_abs[idx * size_platoon] + 10) / 35
    V2V_fast = (env.V2V_channels_with_fastfading[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1)), :]
                - env.V2V_channels_abs[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1))].reshape(size_platoon - 1, 1) + 10) / 35
    Interference = (env.Interference_all[idx] + 60) / 60
    AoI_levels = env.AoI[idx] / n_step_per_episode
    V2V_load_remaining = np.asarray([env.V2V_demand[idx] / env.V2V_demand_size])
    return np.concatenate((np.reshape(V2I_abs, -1), np.reshape(V2I_fast, -1), np.reshape(V2V_abs, -1),
                           np.reshape(V2V_fast, -1), np.reshape(Interference, -1), np.reshape(AoI_levels, -1),
                           V2V_load_remaining), axis=0)


n_input = len(get_state(env, 0))
agents = [IndependentDQNAgent(args.lr, n_input, n_actions, args.gamma, args.fc1, args.fc2,
                              args.batch_size, i, tau=args.target_tau, constraint_mode=args.mode)
          for i in range(n_platoon)]
memory = ReplayBuffer(args.buffer, n_input, n_platoon)

label = 'dqn_' + args.mode + '_seed' + str(SEED)
if args.out_tag:
    label = label + '_' + args.out_tag
if args.out_subdir:
    label = args.out_subdir + '/' + label    # nest all outputs under model/<out_subdir>/
current_dir = os.path.dirname(os.path.realpath(__file__))
print('=== Independent-DQN run: seed=%d episodes=%d steps/ep=%d renew_every=%d n_platoon=%d n_RB=%d '
      'n_input=%d n_actions=%d fc=%d/%d label=%s smoke=%s ==='
      % (SEED, n_episode, n_step_per_episode, args.renew_every, n_platoon, n_RB, n_input, n_actions,
         args.fc1, args.fc2, label, args.smoke))

# ---- logging arrays (mirror the CMDP repo so the same analysis scripts apply) ----
AoI_evolution = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
Demand_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
V2I_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
V2V_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
power_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
AoI_total = np.zeros([n_platoon, n_episode], dtype=np.float16)
rec_r1 = np.zeros([n_platoon, n_episode], dtype=np.float16)
rec_r2 = np.zeros([n_platoon, n_episode], dtype=np.float16)
viol_total = np.zeros([n_platoon, n_episode], dtype=np.float32)     # P(AoI_j > tau) per platoon per episode
eps_total = np.zeros([n_episode], dtype=np.float32)
mode_v2i_total = np.zeros([n_platoon, n_episode], dtype=np.float32) # per-platoon per-episode V2I(inter) mode fraction
lambda_total = np.zeros([n_platoon, n_episode], dtype=np.float32)   # [RQ1-CMDP] per-platoon multiplier (0 throughout in soft)
lam_I = np.zeros(n_platoon, dtype=np.float64)                       # [RQ1-CMDP] PID integral accumulator
lam_err_prev = np.zeros(n_platoon, dtype=np.float64)                # [RQ1-CMDP] PID previous-episode error

total_steps = n_episode * n_step_per_episode
decay_steps = max(1, int(total_steps * args.eps_decay_frac))
global_step = 0


def epsilon_at(step):
    frac = min(1.0, step / decay_steps)
    return args.eps_start + (args.eps_end - args.eps_start) * frac


start = time.time()
for i_episode in range(n_episode):
    env.V2V_demand = env.V2V_demand_size * np.ones(n_platoon, dtype=np.float16)
    env.individual_time_limit = env.time_slow * np.ones(n_platoon, dtype=np.float16)
    env.active_links = np.ones(n_platoon, dtype='bool')
    if i_episode == 0:
        env.AoI = np.ones(n_platoon) * 100                 # AoI carries over thereafter (continuous freshness clock)
    if i_episode % args.renew_every == 0:
        env.renew_positions()
        env.renew_channel(n_veh, size_platoon)
        env.renew_channels_fastfading()

    state_old = [get_state(env, i) for i in range(n_platoon)]
    rec_AoI_step = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)
    rec_r1_step = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)
    rec_r2_step = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)
    rec_mode_step = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)   # mode chosen (0=V2I/inter, 1=V2V/intra)

    for i_step in range(n_step_per_episode):
        eps = epsilon_at(global_step)
        action_idx = np.zeros(n_platoon, dtype=np.int64)
        at = np.zeros([n_platoon, 3], dtype=int)
        for i in range(n_platoon):
            a = agents[i].choose_action(state_old[i], eps)
            action_idx[i] = a
            rb, mode, power = decode_action(a)
            at[i, 0] = rb
            at[i, 1] = mode
            at[i, 2] = power
            rec_mode_step[i, i_step] = mode

        r1, r2, global_reward, platoon_AoI, C_rate, V_rate, Demand_R, V2V_success = env.act_for_training(at.copy())
        cost = (np.asarray(platoon_AoI, dtype=np.float64) > args.tau).astype(np.float32)   # [RQ1-CMDP] per-platoon 1{AoI>tau}
        env.renew_channels_fastfading()
        env.Compute_Interference(at.copy())
        state_new = [get_state(env, i) for i in range(n_platoon)]
        done = (i_step == n_step_per_episode - 1)

        memory.store_transition(np.asarray(state_old).flatten(), action_idx, r1, r2, cost,
                                np.asarray(state_new).flatten(), done)

        if memory.mem_cntr >= args.batch_size:
            s, a_idx, br1, br2, brc, s_, d = memory.sample_buffer(args.batch_size)
            for i in range(n_platoon):
                si = s[:, i * n_input:(i + 1) * n_input]
                si_ = s_[:, i * n_input:(i + 1) * n_input]
                agents[i].learn(si, a_idx[:, i], br1[:, i], br2[:, i], brc[:, i], si_, d)

        for i in range(n_platoon):
            rec_AoI_step[i, i_step] = env.AoI[i]
            rec_r1_step[i, i_step] = r1[i]
            rec_r2_step[i, i_step] = r2[i]
            AoI_evolution[i, i_episode % n_episode_test, i_step] = platoon_AoI[i]
            Demand_total[i, i_episode % n_episode_test, i_step] = Demand_R[i]
            V2I_total[i, i_episode % n_episode_test, i_step] = C_rate[i]
            V2V_total[i, i_episode % n_episode_test, i_step] = V_rate[i]
            power_total[i, i_episode % n_episode_test, i_step] = at[i, 2]

        state_old = state_new
        global_step += 1

    rec_r1[:, i_episode] = rec_r1_step.mean(axis=1)
    rec_r2[:, i_episode] = rec_r2_step.mean(axis=1)
    AoI_total[:, i_episode] = rec_AoI_step.mean(axis=1)
    viol_total[:, i_episode] = (rec_AoI_step > args.tau).mean(axis=1)
    eps_total[i_episode] = eps
    mode_v2i_total[:, i_episode] = 1.0 - rec_mode_step.mean(axis=1)   # fraction of steps in V2I(inter) mode
    # [RQ1-CMDP] two-timescale dual ascent (slow loop, once per episode): raise lambda_j when
    # platoon j's episodic violation exceeds eps, lower it otherwise. Soft mode leaves lambda=0.
    if args.mode == 'hard':
        e_vec = viol_total[:, i_episode] - args.eps
        for j in range(n_platoon):
            e = float(e_vec[j])
            if args.dual == 'integral':
                agents[j].lam = float(np.clip(agents[j].lam + args.eta_lam * e, 0.0, args.lam_max))
            else:  # PID-Lagrangian (Stooke 2020); kp=kd=0, ki=eta_lam reduces to integral
                lam_I[j] = float(np.clip(lam_I[j] + args.ki * e, 0.0, args.lam_max))
                agents[j].lam = float(np.clip(args.kp * e + lam_I[j] + args.kd * (e - lam_err_prev[j]), 0.0, args.lam_max))
                lam_err_prev[j] = e
    lambda_total[:, i_episode] = np.array([agents[j].lam for j in range(n_platoon)])
    print('[dqn %s ep %d] eps=%.3f worst_viol=%.3f net_viol=%.3f meanAoI=%.2f power=%.1f v2v_succ=%.2f v2i_mode=%.2f lam=%s'
          % (args.mode, i_episode, eps, float(viol_total[:, i_episode].max()), float(viol_total[:, i_episode].mean()),
             float(AoI_total[:, i_episode].mean()), float(power_total[:, i_episode % n_episode_test, :].mean()),
             float(V2V_success), float(mode_v2i_total[:, i_episode].mean()), np.round(lambda_total[:, i_episode], 2)))
    if i_episode % 50 == 0:
        for ag in agents:
            ag.save_models()

print('Training done. Saving .mat + checkpoints...')
outdir = os.path.join(current_dir, 'model', label)
os.makedirs(outdir, exist_ok=True)
scipy.io.savemat(os.path.join(outdir, 'reward_t1.mat'), {'reward_t1': rec_r1})
scipy.io.savemat(os.path.join(outdir, 'reward_t2.mat'), {'reward_t2': rec_r2})
scipy.io.savemat(os.path.join(outdir, 'AoI.mat'), {'AoI': AoI_total})
scipy.io.savemat(os.path.join(outdir, 'viol_rate.mat'), {'viol_rate': viol_total})
scipy.io.savemat(os.path.join(outdir, 'AoI_evolution.mat'), {'AoI_evolution': AoI_evolution})
scipy.io.savemat(os.path.join(outdir, 'demand.mat'), {'demand': Demand_total})
scipy.io.savemat(os.path.join(outdir, 'V2I.mat'), {'V2I': V2I_total})
scipy.io.savemat(os.path.join(outdir, 'V2V.mat'), {'V2V': V2V_total})
scipy.io.savemat(os.path.join(outdir, 'power.mat'), {'power': power_total})
scipy.io.savemat(os.path.join(outdir, 'epsilon.mat'), {'epsilon': eps_total})
scipy.io.savemat(os.path.join(outdir, 'mode_v2i.mat'), {'mode_v2i': mode_v2i_total})
scipy.io.savemat(os.path.join(outdir, 'lambda.mat'), {'lambda': lambda_total})   # [RQ1-CMDP] per-platoon multiplier
for ag in agents:
    ag.save_models()
print('done. label=%s  time=%.1fs' % (label, time.time() - start))
