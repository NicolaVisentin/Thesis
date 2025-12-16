# =====================================================
# Setup
# =====================================================

# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Imports
import sys
from pathlib import Path
import time

import numpy as onp
import jax
import jax.numpy as jnp
from diffrax import Tsit5, ConstantStepSize

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.widgets import Slider

from soromox.systems.my_systems import PlanarPCS_simple
from soromox.systems.system_state import SystemState

curr_folder = Path(__file__).parent      # current folder
sys.path.append(str(curr_folder.parent)) # scripts folder
from utilis import *

# Jax settings
jax.config.update("jax_enable_x64", True)  # double precision
jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)
seed = 123
key = jax.random.key(seed)

# Folders
main_folder = curr_folder.parent.parent                                            # main folder "codes"
plots_folder = main_folder/'plots and videos'/curr_folder.stem/Path(__file__).stem # folder for plots and videos
dataset_folder = main_folder/'datasets'                                            # folder with the dataset
data_folder = main_folder/'saved data/ablation_study'                              # folder with saved data

plots_folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# Script settings
# =====================================================
do_ref_case = True
do_nopcs_case = True
do_nomap_case = True
do_diagmap_case = True
do_nomlp_case = True
do_regulmlp_case = True
do_regulmap_case = True
do_coupled_case = True
do_overall = True


# =====================================================
# Utilis
# =====================================================

# Functions for plotting robot
def draw_robot(
        robot: PlanarPCS_simple, 
        q: Array, 
        num_points: int = 50
):
    L_max = jnp.sum(robot.L)
    s_ps = jnp.linspace(0, L_max, num_points)

    chi_ps = robot.forward_kinematics_batched(q, s_ps)  # (N,3)
    curve = jnp.array(chi_ps[:, 1:], dtype=jnp.float64) # (N,2)
    pos_tip = curve[-1]                                 # [x_tip, y_tip]

    return curve, pos_tip

def animate_robot_matplotlib(
    robot: PlanarPCS_simple,
    t_list: Array,  # shape (T,)
    q_list: Array,  # shape (T, DOF)
    target: Array = None,
    num_points: int = 50,
    interval: int = 50,
    slider: bool = None,
    animation: bool = None,
    show: bool = True,
    fps: float = None,
    duration: float = None,
    save_path: Path = None,
):
    if slider is None and animation is None:
        raise ValueError("Either 'slider' or 'animation' must be set to True.")
    if animation and slider:
        raise ValueError(
            "Cannot use both animation and slider at the same time. Choose one."
        )

    _, pos_tip_list = jax.vmap(draw_robot, in_axes=(None,0,None))(robot, q_list, num_points)
    width = onp.max(pos_tip_list)
    height = width

    if target is not None:
        t_old = onp.linspace(0, 1, len(target))
        t_new = onp.linspace(0, 1, len(q_list))
        target = onp.interp(t_new, t_old, target)

    def draw_base(ax, robot, L=robot.L[0] / 2):
        angle1 = robot.th0 - jnp.pi / 2
        angle2 = robot.th0 + jnp.pi / 2
        x1, y1 = L * jnp.cos(angle1), L * jnp.sin(angle1)
        x2, y2 = L * jnp.cos(angle2), L * jnp.sin(angle2)
        ax.plot([x1, x2], [y1, y2], color="black", linestyle="-", linewidth=2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_base(ax, robot, L=0.1)

    if animation:
        n_frames = len(q_list)
        default_fps = 1000 / interval
        default_duration = n_frames * (interval / 1000)
        if fps is None and duration is None:
            final_fps = default_fps
            final_duration = default_duration
            frame_skip = 1
        elif fps is not None and duration is None:
            final_fps = fps
            final_duration = n_frames / save_fps
            frame_skip = 1
        elif fps is None and duration is not None:
            final_duration = duration
            final_fps = n_frames / duration
            frame_skip = 1
        else:
            final_fps = fps
            final_duration = duration
            n_required_frames = int(final_duration * final_fps)
            n_required_frames = max(1, n_required_frames)
            frame_skip = max(1, n_frames // n_required_frames)

        frame_indices = list(range(0, n_frames, frame_skip))

        (line,) = ax.plot([], [], lw=4, color="blue")
        (tip,) = ax.plot([], [], 'ro', markersize=5)
        (targ,) = ax.plot([], [], color='r', alpha=0.5)
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(0, height)
        ax.grid(True)
        title_text = ax.set_title("t = 0.00 s")

        def init():
            line.set_data([], [])
            tip.set_data([], [])
            targ.set_data([], [])
            title_text.set_text("t = 0.00 s")
            return line, tip, targ, title_text

        def update(frame_idx):
            q = q_list[frame_idx]
            t = t_list[frame_idx]
            curve, tip_pos = draw_robot(robot, q, num_points)
            line.set_data(curve[:, 0], curve[:, 1])
            tip.set_data([tip_pos[0]], [tip_pos[1]])
            if target is not None:
                x_target = target[frame_idx]
                targ.set_data([x_target,x_target], [0,height])
            title_text.set_text(f"t = {t:.2f} s")
            return (line, tip, title_text) + ((targ,) if target is not None else ())

        ani = FuncAnimation(
            fig,
            update,
            frames=frame_indices,
            init_func=init,
            blit=False,
            interval=1000/final_fps,
        )
        if save_path is not None:
            ani.save(save_path, writer=PillowWriter(fps=final_fps))

    elif slider:

        def update_plot(frame_idx):
            ax.cla()  # Clear current axes
            ax.set_xlim(-width / 2, width / 2)
            ax.set_ylim(0, height)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title(f"t = {t_list[frame_idx]:.2f} s")
            ax.grid(True)
            q = q_list[frame_idx]
            curve, tip_pos = draw_robot(robot, q, num_points)
            ax.plot(curve[:, 0], curve[:, 1], lw=4, color="blue")
            ax.plot([tip_pos[0]], [tip_pos[1]], 'ro', markersize=5)
            if target is not None:
                x_target = target[frame_idx]
                ax.plot([x_target,x_target], [0,height], 'r', alpha=0.5)
            fig.canvas.draw_idle()

        # Create slider
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]
        slider = Slider(
            ax=ax_slider,
            label="Frame",
            valmin=0,
            valmax=len(t_list) - 1,
            valinit=0,
            valstep=1,
        )
        slider.on_changed(update_plot)

        update_plot(0)  # Initial plot

    if show:
        plt.show()

    plt.close(fig)

# Dummy instantiations
n_pcs = 2
parameters = {
    "th0": jnp.array(jnp.pi/2),
    "L": jnp.ones(n_pcs),
    "r": jnp.ones(n_pcs),
    "rho": jnp.ones(n_pcs),
    "g": jnp.array([0.0, 9.81]), # !! gravity UP !!
    "E": jnp.ones(n_pcs),
    "G": jnp.ones(n_pcs),
    "D": jnp.eye(3*n_pcs)
}
robot = PlanarPCS_simple(
    num_segments = n_pcs,
    params = parameters,
    order_gauss = 5
)

key, subkey = jax.random.split(key)
mlp_controller = MLP(key=subkey, layer_sizes=[2*3*n_pcs, 64, 64, 3*n_pcs])

def tau_law(system_state: SystemState, controller: MLP):
    """Implements user-defined feedback control tau(t) = MLP(q(t),qd(t))."""
    tau = controller(system_state.y)
    return tau, None

def map(r, A, c):
    y, yd = jnp.split(r, 2)
    q = A @ y + c
    qd = A @ yd
    z = jnp.concatenate([q, qd])
    return z

# RON dataset (decoupled case)
RON_dataset = onp.load(dataset_folder/'soft robot optimization/N6_simplified/dataset_m1e5_N6_simplified.npz')

# RON data (decoupled case)
RON_evolution_data = onp.load(dataset_folder/'soft robot optimization/N6_simplified/RON_evolution_N6_simplified_a.npz')
time_RONsaved = jnp.array(RON_evolution_data['time'])
y_RONsaved = jnp.array(RON_evolution_data['y'])
yd_RONsaved = jnp.array(RON_evolution_data['yd'])

# RON data (coupled case)
RON_evolution_data_coupled = onp.load(dataset_folder/'soft robot optimization/N6_noInput/RON_evolution_N6_noInput.npz')
time_RONsaved_coupled = jnp.array(RON_evolution_data_coupled['time'])
y_RONsaved_coupled = jnp.array(RON_evolution_data_coupled['y'])
yd_RONsaved_coupled = jnp.array(RON_evolution_data_coupled['yd'])

# Simulation parameters
t0 = time_RONsaved[0]
t1 = time_RONsaved[-1]
dt = 1e-4
saveat = np.arange(t0, t1, 1e-2)
solver = Tsit5()
step_size = ConstantStepSize()
max_steps = int(1e6)


# =====================================================
# 0.0 Reference case
# =====================================================
if do_ref_case or do_overall:
    print(f'--- REFERENCE CASE ---')
    test_case = '0.0_reference'
    (plots_folder/test_case).mkdir(parents=True, exist_ok=True)

    ##### ALL SAMPLES #####
    prefix = 'SAMPLES_REF'

    # Load and extract data
    all_loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    all_rmse_before = onp.load(data_folder/test_case/f'{prefix}_all_rmse_before.npz')
    all_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')
    all_robot_params_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    all_robot_params_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    all_map_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    all_powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')

    all_train_loss_ts = all_loss_curves["train_losses_ts"]
    all_val_loss_ts = all_loss_curves["val_losses_ts"]
    all_train_mse_ts = all_loss_curves["train_MSEs_ts"]
    all_rmse_before = all_rmse_before["RMSE_before"]
    SAMPLES_REF_all_rmse_after = all_rmse_after["RMSE_after"]
    n_samples = all_rmse_before.shape[0]
    SAMPLES_REF_all_powers_msv_after = all_powers_msv_after["powers_msv_after"]
    n_epochs_samples = all_train_mse_ts.shape[1]

    # Compute "mapping effort" for each sample (after training)
    print('Computing mapping effort (after training)')
    SAMPLES_REF_mapping_effort_after = []
    for i in range(n_samples):
        robot_i = robot.update_params({
            "L": jnp.array(all_robot_params_after["L_after"][i]), 
            "D": jnp.diag(all_robot_params_after["D_after"][i]),
            "r": jnp.array(all_robot_params_after["r_after"][i]),
            "rho": jnp.array(all_robot_params_after["rho_after"][i]),
            "E": jnp.array(all_robot_params_after["E_after"][i]),
            "G": jnp.array(all_robot_params_after["G_after"][i]),
        })
        map_i = partial(map, A=jnp.array(all_map_after["A_after"][i]), c=jnp.array(all_map_after["c_after"][i]))
        mapping_effort_i = mean_Ek_ratio(robot_i, RON_dataset, map_i)
        SAMPLES_REF_mapping_effort_after.append(mapping_effort_i)

    if do_ref_case:
        # Plot comparison of all samples (RMSE)
        colors = plt.cm.viridis(onp.linspace(0,1,n_samples))

        plt.figure()
        plt.scatter(onp.arange(n_samples)+1, all_rmse_before, marker='x', c=colors, label='test RMSE before')
        plt.scatter(onp.arange(n_samples)+1, SAMPLES_REF_all_rmse_after, marker='o', c=colors, label='test RMSE after')
        plt.scatter(onp.arange(n_samples)+1, onp.sqrt(all_train_mse_ts[:,-1]), marker='+', c=colors, label='final train RMSE')
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('sample n.')
        plt.ylabel('RMSE')
        plt.title(f'Results for various initial guesses')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_comparison', bbox_inches='tight')
        #plt.show()

        # Plot comparison of all samples (loss curves)
        plt.figure()
        for i in range(n_samples):
            plt.plot(range(n_epochs_samples), all_train_loss_ts[i], color=colors[i], label=f'train losses' if i == 0 else "")
            plt.plot(onp.arange(1, n_epochs_samples + 1), all_val_loss_ts[i], '--', color=colors[i], label=f'validation losses' if i == 0 else "")
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Results for all samples')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_losses', bbox_inches='tight')
        #plt.show()

        # Save text file with all initial and final pcs parameters for the robot
        with open(plots_folder/test_case/'samples_pcs_params_comparison.txt', 'w') as file:
            file.write(f'PCS parameters before and after training for all samples:\n\n')
            for i in range(n_samples):
                file.write(f'L = {all_robot_params_before["L_before"][i]} --> {all_robot_params_after["L_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'D = {all_robot_params_before["D_before"][i]} --> {all_robot_params_after["D_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'r = {all_robot_params_before["r_before"][i]} --> {all_robot_params_after["r_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'rho = {all_robot_params_before["rho_before"][i]} --> {all_robot_params_after["rho_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'E = {all_robot_params_before["E_before"][i]} --> {all_robot_params_after["E_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'G = {all_robot_params_before["G_before"][i]} --> {all_robot_params_after["G_after"][i]}\n')

    ##### BEST RESULT #####
    prefix = 'BEST_REF'

    # Load and extract data (training)
    loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    REF_train_loss_ts = loss_curves["train_losses_ts"][0]
    REF_val_loss_ts = loss_curves["val_losses_ts"][0]
    n_epochs = len(REF_train_loss_ts)

    # Load and extract data (before training)
    CONTR_before = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_before.npz')
    CONTR_before = mlp_controller.extract_params_from_batch(CONTR_before, 0) # controller data are always saved as batches
    controller_before = mlp_controller.update_params(CONTR_before)
    powers_msv_before = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_before.npz')
    powers_msv_before = powers_msv_before["powers_msv_before"][0]

    robot_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    L_before = jnp.array(robot_data_before["L_before"][0])
    D_before = jnp.array(robot_data_before["D_before"][0])
    r_before = jnp.array(robot_data_before["r_before"][0])
    rho_before = jnp.array(robot_data_before["rho_before"][0])
    E_before = jnp.array(robot_data_before["E_before"][0])
    G_before = jnp.array(robot_data_before["G_before"][0])
    robot_before = robot.update_params({"L": L_before, "D": jnp.diag(D_before), "r": r_before, "rho": rho_before, "E": E_before, "G": G_before})

    map_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_map_before.npz')
    A_before = jnp.array(map_data_before["A_before"][0])
    c_before = jnp.array(map_data_before["c_before"][0])

    # Load and extract data (after training)
    BEST_REF_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')["RMSE_after"][0]

    CONTR_after = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_after.npz')
    CONTR_after = mlp_controller.extract_params_from_batch(CONTR_after, 0) # controller data are always saved as batches
    controller_after = mlp_controller.update_params(CONTR_after)
    powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')
    BEST_REF_powers_msv_after = powers_msv_after["powers_msv_after"][0]

    robot_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    L_after = jnp.array(robot_data_after["L_after"][0])
    D_after = jnp.array(robot_data_after["D_after"][0])
    r_after = jnp.array(robot_data_after["r_after"][0])
    rho_after = jnp.array(robot_data_after["rho_after"][0])
    E_after = jnp.array(robot_data_after["E_after"][0])
    G_after = jnp.array(robot_data_after["G_after"][0])
    robot_after = robot.update_params({"L": L_after, "D": jnp.diag(D_after), "r": r_after, "rho": rho_after, "E": E_after, "G": G_after})

    map_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    A_after = jnp.array(map_data_after["A_after"][0])
    c_after = jnp.array(map_data_after["c_after"][0])

    # Compute mapping effort (after training)
    BEST_REF_mapping_effort_after = mean_Ek_ratio(
        robot_after, 
        RON_dataset, 
        partial(map, A=A_after, c=c_after)
    )
    BEST_REF_condAinv = jnp.linalg.cond(jnp.linalg.inv(A_after))

    if do_ref_case:
        # Simulation before training
        print('Simulating best case (before training)...')
        q0 = A_before @ y_RONsaved[0] + c_before
        qd0 = A_before @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_before)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_before.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_before = sim_out_pcs.t
        q_PCS_before, qd_PCS_before = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_before = sim_out_pcs.u
        y_hat_pcs_before = jnp.linalg.solve(A_before, (q_PCS_before - c_before).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_before = jnp.linalg.solve(A_before, qd_PCS_before.T).T            # yd_hat(t) = inv(A) * qd(t)

        # Simulation after training
        print('Simulating best case (after training)...')
        q0 = A_after @ y_RONsaved[0] + c_after
        qd0 = A_after @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_after)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_after.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_after = sim_out_pcs.t
        q_PCS_after, qd_PCS_after = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_after = sim_out_pcs.u
        y_hat_pcs_after = jnp.linalg.solve(A_after, (q_PCS_after - c_after).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_after = jnp.linalg.solve(A_after, qd_PCS_after.T).T           # yd_hat(t) = inv(A) * qd(t)

        # Compute simulation metrics
        BEST_REF_simulationPower = compute_simulation_power(u_pcs_after, qd_PCS_after)
        BEST_REF_simulationAccuracy = compute_simulation_rmse(timePCS_after, y_hat_pcs_after, time_RONsaved, y_RONsaved)
        BEST_REF_simulationMapeffort = compute_simulation_Ek_ratio(robot_after, timePCS_after, q_PCS_after, qd_PCS_after, time_RONsaved, yd_RONsaved)

        # Show loss curve
        plt.figure()
        plt.plot(range(n_epochs), REF_train_loss_ts, 'r', label='train loss')
        plt.plot(onp.arange(1,n_epochs+1), REF_val_loss_ts, 'b', label='validation loss')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Result for best initial guess')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_loss', bbox_inches='tight')
        #plt.show()

        # Show and save animation before training 
        animate_robot_matplotlib(
            robot = robot_before,
            t_list = saveat,
            q_list = q_PCS_before,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_before.gif',
        )

        # Show animation after training
        animate_robot_matplotlib(
            robot = robot_after,
            t_list = saveat,
            q_list = q_PCS_after,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_after.gif',
        )

        # Plot robot strains and control torque before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_before, q_PCS_before[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_before, q_PCS_before[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_before, q_PCS_before[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_before, u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_before, u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_before, u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_before', bbox_inches='tight')
        #plt.show()

        # Plot robot strains and control torque after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_after, q_PCS_after[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_after, q_PCS_after[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_after, q_PCS_after[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_after, u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_after, u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_after, u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_after', bbox_inches='tight')
        #plt.show()

        # Plot actuation power before training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_before, qd_PCS_before[:,i] * u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_before, qd_PCS_before[:,i+1] * u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_before, qd_PCS_before[:,i+2] * u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_before', bbox_inches='tight')
        #plt.show()

        # Plot actuation power after training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_after, qd_PCS_after[:,i] * u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_after, qd_PCS_after[:,i+1] * u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_after, qd_PCS_after[:,i+2] * u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_after', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_before, y_hat_pcs_before[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_before', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_after, y_hat_pcs_after[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_after', bbox_inches='tight')
        #plt.show()

        # Plot phase planes before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_before[:, i], yd_hat_pcs_before[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_before', bbox_inches='tight')
        #plt.show()

        # Plot phase planes after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_after[:, i], yd_hat_pcs_after[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_after', bbox_inches='tight')
        #plt.show()

        # Save in a text file all the parameters before and after training
        with open(plots_folder/test_case/'best_result_parameters.txt', 'w') as file:
            file.write(f'----------BEFORE TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_before}\n')
            file.write(f'D = {D_before}\n')
            file.write(f'r = {r_before}\n')
            file.write(f'rho = {rho_before}\n')
            file.write(f'E = {E_before}\n')
            file.write(f'G = {G_before}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_before}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_before)}\n')
            file.write(f'c = {c_before}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(powers_msv_before)}\n')
            file.write(f'\n\n----------AFTER TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_after}\n')
            file.write(f'D = {D_after}\n')
            file.write(f'r = {r_after}\n')
            file.write(f'rho = {rho_after}\n')
            file.write(f'E = {E_after}\n')
            file.write(f'G = {G_after}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_after}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_after)}\n')
            file.write(f'c = {c_after}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(BEST_REF_powers_msv_after)}\n')


#plt.show()
plt.close() # close figures to free memory
# =====================================================
# 1.1 No PCS
# =====================================================
if do_nopcs_case or do_overall:
    print(f'\n--- NO PCS CASE ---')
    test_case = '1.1_noPCS'
    (plots_folder/test_case).mkdir(parents=True, exist_ok=True)

    ##### ALL SAMPLES #####
    prefix = 'SAMPLES_NOPCS'

    # Load and extract data
    all_loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    all_rmse_before = onp.load(data_folder/test_case/f'{prefix}_all_rmse_before.npz')
    all_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')
    all_robot_params_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    all_robot_params_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    all_map_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    all_powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')

    all_train_loss_ts = all_loss_curves["train_losses_ts"]
    all_val_loss_ts = all_loss_curves["val_losses_ts"]
    all_train_mse_ts = all_loss_curves["train_MSEs_ts"]
    all_rmse_before = all_rmse_before["RMSE_before"]
    SAMPLES_NOPCS_all_rmse_after = all_rmse_after["RMSE_after"]
    n_samples = all_rmse_before.shape[0]
    SAMPLES_NOPCS_all_powers_msv_after = all_powers_msv_after["powers_msv_after"]
    n_epochs_samples = all_train_mse_ts.shape[1]

    # Compute "mapping effort" for each sample (after training)
    print('Computing mapping effort (after training)')
    SAMPLES_NOPCS_mapping_effort_after = []
    for i in range(n_samples):
        robot_i = robot.update_params({
            "L": jnp.array(all_robot_params_after["L_after"][i]), 
            "D": jnp.diag(all_robot_params_after["D_after"][i]),
            "r": jnp.array(all_robot_params_after["r_after"][i]),
            "rho": jnp.array(all_robot_params_after["rho_after"][i]),
            "E": jnp.array(all_robot_params_after["E_after"][i]),
            "G": jnp.array(all_robot_params_after["G_after"][i]),
        })
        map_i = partial(map, A=jnp.array(all_map_after["A_after"][i]), c=jnp.array(all_map_after["c_after"][i]))
        mapping_effort_i = mean_Ek_ratio(robot_i, RON_dataset, map_i)
        SAMPLES_NOPCS_mapping_effort_after.append(mapping_effort_i)

    if do_nopcs_case:
        # Plot comparison of all samples (RMSE)
        colors = plt.cm.viridis(onp.linspace(0,1,n_samples))

        plt.figure()
        plt.scatter(onp.arange(n_samples)+1, all_rmse_before, marker='x', c=colors, label='test RMSE before')
        plt.scatter(onp.arange(n_samples)+1, SAMPLES_NOPCS_all_rmse_after, marker='o', c=colors, label='test RMSE after')
        plt.scatter(onp.arange(n_samples)+1, onp.sqrt(all_train_mse_ts[:,-1]), marker='+', c=colors, label='final train RMSE')
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('sample n.')
        plt.ylabel('RMSE')
        plt.title(f'Results for various initial guesses')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_comparison', bbox_inches='tight')
        #plt.show()

        # Plot comparison of all samples (loss curves)
        plt.figure()
        for i in range(n_samples):
            plt.plot(range(n_epochs_samples), all_train_loss_ts[i], color=colors[i], label=f'train losses' if i == 0 else "")
            plt.plot(onp.arange(1, n_epochs_samples + 1), all_val_loss_ts[i], '--', color=colors[i], label=f'validation losses' if i == 0 else "")
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Results for all samples')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_losses', bbox_inches='tight')
        #plt.show()

        # Save text file with all initial and final pcs parameters for the robot
        with open(plots_folder/test_case/'samples_pcs_params_comparison.txt', 'w') as file:
            file.write(f'PCS parameters before and after training for all samples:\n\n')
            for i in range(n_samples):
                file.write(f'L = {all_robot_params_before["L_before"][i]} --> {all_robot_params_after["L_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'D = {all_robot_params_before["D_before"][i]} --> {all_robot_params_after["D_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'r = {all_robot_params_before["r_before"][i]} --> {all_robot_params_after["r_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'rho = {all_robot_params_before["rho_before"][i]} --> {all_robot_params_after["rho_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'E = {all_robot_params_before["E_before"][i]} --> {all_robot_params_after["E_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'G = {all_robot_params_before["G_before"][i]} --> {all_robot_params_after["G_after"][i]}\n')

    ##### BEST RESULT #####
    prefix = 'BEST_NOPCS' # !! In this test, BEST_NOPCS is taking as initial guess the best REFERENCE intial guess, to make a direct comparison

    # Load and extract data (training)
    loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    NOPCS_train_loss_ts = loss_curves["train_losses_ts"][0]
    NOPCS_val_loss_ts = loss_curves["val_losses_ts"][0]
    n_epochs = len(NOPCS_train_loss_ts)

    # Load and extract data (before training)
    CONTR_before = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_before.npz')
    CONTR_before = mlp_controller.extract_params_from_batch(CONTR_before, 0) # controller data are always saved as batches
    controller_before = mlp_controller.update_params(CONTR_before)
    powers_msv_before = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_before.npz')
    powers_msv_before = powers_msv_before["powers_msv_before"][0]

    robot_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    L_before = jnp.array(robot_data_before["L_before"][0])
    D_before = jnp.array(robot_data_before["D_before"][0])
    r_before = jnp.array(robot_data_before["r_before"][0])
    rho_before = jnp.array(robot_data_before["rho_before"][0])
    E_before = jnp.array(robot_data_before["E_before"][0])
    G_before = jnp.array(robot_data_before["G_before"][0])
    robot_before = robot.update_params({"L": L_before, "D": jnp.diag(D_before), "r": r_before, "rho": rho_before, "E": E_before, "G": G_before})

    map_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_map_before.npz')
    A_before = jnp.array(map_data_before["A_before"][0])
    c_before = jnp.array(map_data_before["c_before"][0])

    # Load and extract data (after training)
    BEST_NOPCS_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')["RMSE_after"][0]

    CONTR_after = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_after.npz')
    CONTR_after = mlp_controller.extract_params_from_batch(CONTR_after, 0) # controller data are always saved as batches
    controller_after = mlp_controller.update_params(CONTR_after)
    powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')
    BEST_NOPCS_powers_msv_after = powers_msv_after["powers_msv_after"][0]

    robot_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    L_after = jnp.array(robot_data_after["L_after"][0])
    D_after = jnp.array(robot_data_after["D_after"][0])
    r_after = jnp.array(robot_data_after["r_after"][0])
    rho_after = jnp.array(robot_data_after["rho_after"][0])
    E_after = jnp.array(robot_data_after["E_after"][0])
    G_after = jnp.array(robot_data_after["G_after"][0])
    robot_after = robot.update_params({"L": L_after, "D": jnp.diag(D_after), "r": r_after, "rho": rho_after, "E": E_after, "G": G_after})

    map_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    A_after = jnp.array(map_data_after["A_after"][0])
    c_after = jnp.array(map_data_after["c_after"][0])

    # Compute mapping effort (after training)
    BEST_NOPCS_mapping_effort_after = mean_Ek_ratio(
        robot_after, 
        RON_dataset, 
        partial(map, A=A_after, c=c_after)
    )
    BEST_NOPCS_condAinv = jnp.linalg.cond(jnp.linalg.inv(A_after))

    if do_nopcs_case:
        # Simulation before training
        print('Simulating best case (before training)...')
        q0 = A_before @ y_RONsaved[0] + c_before
        qd0 = A_before @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_before)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_before.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_before = sim_out_pcs.t
        q_PCS_before, qd_PCS_before = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_before = sim_out_pcs.u
        y_hat_pcs_before = jnp.linalg.solve(A_before, (q_PCS_before - c_before).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_before = jnp.linalg.solve(A_before, qd_PCS_before.T).T            # yd_hat(t) = inv(A) * qd(t)

        # Simulation after training
        print('Simulating best case (after training)...')
        q0 = A_after @ y_RONsaved[0] + c_after
        qd0 = A_after @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_after)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_after.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_after = sim_out_pcs.t
        q_PCS_after, qd_PCS_after = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_after = sim_out_pcs.u
        y_hat_pcs_after = jnp.linalg.solve(A_after, (q_PCS_after - c_after).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_after = jnp.linalg.solve(A_after, qd_PCS_after.T).T           # yd_hat(t) = inv(A) * qd(t)

        # Compute simulation metrics
        BEST_NOPCS_simulationPower = compute_simulation_power(u_pcs_after, qd_PCS_after)
        BEST_NOPCS_simulationAccuracy = compute_simulation_rmse(timePCS_after, y_hat_pcs_after, time_RONsaved, y_RONsaved)
        BEST_NOPCS_simulationMapeffort = compute_simulation_Ek_ratio(robot_after, timePCS_after, q_PCS_after, qd_PCS_after, time_RONsaved, yd_RONsaved)

        # Show loss curve
        plt.figure()
        plt.plot(range(n_epochs), NOPCS_train_loss_ts, 'r', label='train loss')
        plt.plot(onp.arange(1,n_epochs+1), NOPCS_val_loss_ts, 'b', label='validation loss')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Result for best initial guess')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_loss', bbox_inches='tight')
        #plt.show()

        # Show and save animation before training 
        animate_robot_matplotlib(
            robot = robot_before,
            t_list = saveat,
            q_list = q_PCS_before,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_before.gif',
        )

        # Show animation after training
        animate_robot_matplotlib(
            robot = robot_after,
            t_list = saveat,
            q_list = q_PCS_after,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_after.gif',
        )

        # Plot robot strains and control torque before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_before, q_PCS_before[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_before, q_PCS_before[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_before, q_PCS_before[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_before, u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_before, u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_before, u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_before', bbox_inches='tight')
        #plt.show()

        # Plot robot strains and control torque after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_after, q_PCS_after[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_after, q_PCS_after[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_after, q_PCS_after[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_after, u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_after, u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_after, u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_after', bbox_inches='tight')
        #plt.show()

        # Plot actuation power before training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_before, qd_PCS_before[:,i] * u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_before, qd_PCS_before[:,i+1] * u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_before, qd_PCS_before[:,i+2] * u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_before', bbox_inches='tight')
        #plt.show()

        # Plot actuation power after training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_after, qd_PCS_after[:,i] * u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_after, qd_PCS_after[:,i+1] * u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_after, qd_PCS_after[:,i+2] * u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_after', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_before, y_hat_pcs_before[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_before', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_after, y_hat_pcs_after[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_after', bbox_inches='tight')
        #plt.show()

        # Plot phase planes before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_before[:, i], yd_hat_pcs_before[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_before', bbox_inches='tight')
        #plt.show()

        # Plot phase planes after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_after[:, i], yd_hat_pcs_after[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_after', bbox_inches='tight')
        #plt.show()

        # Save in a text file all the parameters before and after training
        with open(plots_folder/test_case/'best_result_parameters.txt', 'w') as file:
            file.write(f'*Important note: "best_nopcs" case (i.e. long optimization of the "noPCS" case) takes as initial guesses for all parameters the best initial guess of the REFERENCE case, not of the noPCS case.\n\n')
            file.write(f'----------BEFORE TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_before}\n')
            file.write(f'D = {D_before}\n')
            file.write(f'r = {r_before}\n')
            file.write(f'rho = {rho_before}\n')
            file.write(f'E = {E_before}\n')
            file.write(f'G = {G_before}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_before}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_before)}\n')
            file.write(f'c = {c_before}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(powers_msv_before)}\n')
            file.write(f'\n\n----------AFTER TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_after}\n')
            file.write(f'D = {D_after}\n')
            file.write(f'r = {r_after}\n')
            file.write(f'rho = {rho_after}\n')
            file.write(f'E = {E_after}\n')
            file.write(f'G = {G_after}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_after}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_after)}\n')
            file.write(f'c = {c_after}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(BEST_NOPCS_powers_msv_after)}\n')


#plt.show()
plt.close() # close figures to free memory
# =====================================================
# 2.1 No mapping (identity) case
# =====================================================
if do_nomap_case or do_overall:
    print(f'\n--- NO MAP CASE ---')
    test_case = '2.1_noMap'
    (plots_folder/test_case).mkdir(parents=True, exist_ok=True)

    ##### ALL SAMPLES #####
    prefix = 'SAMPLES_NOMAP'

    # Load and extract data
    all_loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    all_rmse_before = onp.load(data_folder/test_case/f'{prefix}_all_rmse_before.npz')
    all_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')
    all_robot_params_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    all_robot_params_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    all_map_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    all_powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')

    all_train_loss_ts = all_loss_curves["train_losses_ts"]
    all_val_loss_ts = all_loss_curves["val_losses_ts"]
    all_train_mse_ts = all_loss_curves["train_MSEs_ts"]
    all_rmse_before = all_rmse_before["RMSE_before"]
    SAMPLES_NOMAP_all_rmse_after = all_rmse_after["RMSE_after"]
    n_samples = all_rmse_before.shape[0]
    SAMPLES_NOMAP_all_powers_msv_after = all_powers_msv_after["powers_msv_after"]
    n_epochs_samples = all_train_mse_ts.shape[1]

    # Compute "mapping effort" for each sample (after training)
    print('Computing mapping effort (after training)')
    SAMPLES_NOMAP_mapping_effort_after = []
    for i in range(n_samples):
        robot_i = robot.update_params({
            "L": jnp.array(all_robot_params_after["L_after"][i]), 
            "D": jnp.diag(all_robot_params_after["D_after"][i]),
            "r": jnp.array(all_robot_params_after["r_after"][i]),
            "rho": jnp.array(all_robot_params_after["rho_after"][i]),
            "E": jnp.array(all_robot_params_after["E_after"][i]),
            "G": jnp.array(all_robot_params_after["G_after"][i]),
        })
        map_i = partial(map, A=jnp.array(all_map_after["A_after"][i]), c=jnp.array(all_map_after["c_after"][i]))
        mapping_effort_i = mean_Ek_ratio(robot_i, RON_dataset, map_i)
        SAMPLES_NOMAP_mapping_effort_after.append(mapping_effort_i)

    if do_nomap_case:
        # Plot comparison of all samples (RMSE)
        colors = plt.cm.viridis(onp.linspace(0,1,n_samples))

        plt.figure()
        plt.scatter(onp.arange(n_samples)+1, all_rmse_before, marker='x', c=colors, label='test RMSE before')
        plt.scatter(onp.arange(n_samples)+1, SAMPLES_NOMAP_all_rmse_after, marker='o', c=colors, label='test RMSE after')
        plt.scatter(onp.arange(n_samples)+1, onp.sqrt(all_train_mse_ts[:,-1]), marker='+', c=colors, label='final train RMSE')
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('sample n.')
        plt.ylabel('RMSE')
        plt.title(f'Results for various initial guesses')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_comparison', bbox_inches='tight')
        #plt.show()

        # Plot comparison of all samples (loss curves)
        plt.figure()
        for i in range(n_samples):
            plt.plot(range(n_epochs_samples), all_train_loss_ts[i], color=colors[i], label=f'train losses' if i == 0 else "")
            plt.plot(onp.arange(1, n_epochs_samples + 1), all_val_loss_ts[i], '--', color=colors[i], label=f'validation losses' if i == 0 else "")
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Results for all samples')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_losses', bbox_inches='tight')
        #plt.show()

        # Save text file with all initial and final pcs parameters for the robot
        with open(plots_folder/test_case/'samples_pcs_params_comparison.txt', 'w') as file:
            file.write(f'PCS parameters before and after training for all samples:\n\n')
            for i in range(n_samples):
                file.write(f'L = {all_robot_params_before["L_before"][i]} --> {all_robot_params_after["L_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'D = {all_robot_params_before["D_before"][i]} --> {all_robot_params_after["D_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'r = {all_robot_params_before["r_before"][i]} --> {all_robot_params_after["r_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'rho = {all_robot_params_before["rho_before"][i]} --> {all_robot_params_after["rho_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'E = {all_robot_params_before["E_before"][i]} --> {all_robot_params_after["E_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'G = {all_robot_params_before["G_before"][i]} --> {all_robot_params_after["G_after"][i]}\n')

    ##### BEST RESULT #####
    prefix = 'BEST_NOMAP'

    # Load and extract data (training)
    loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    NOMAP_train_loss_ts = loss_curves["train_losses_ts"][0]
    NOMAP_val_loss_ts = loss_curves["val_losses_ts"][0]
    n_epochs = len(NOMAP_train_loss_ts)

    # Load and extract data (before training)
    CONTR_before = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_before.npz')
    CONTR_before = mlp_controller.extract_params_from_batch(CONTR_before, 0) # controller data are always saved as batches
    controller_before = mlp_controller.update_params(CONTR_before)
    powers_msv_before = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_before.npz')
    powers_msv_before = powers_msv_before["powers_msv_before"][0]

    robot_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    L_before = jnp.array(robot_data_before["L_before"][0])
    D_before = jnp.array(robot_data_before["D_before"][0])
    r_before = jnp.array(robot_data_before["r_before"][0])
    rho_before = jnp.array(robot_data_before["rho_before"][0])
    E_before = jnp.array(robot_data_before["E_before"][0])
    G_before = jnp.array(robot_data_before["G_before"][0])
    robot_before = robot.update_params({"L": L_before, "D": jnp.diag(D_before), "r": r_before, "rho": rho_before, "E": E_before, "G": G_before})

    map_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_map_before.npz')
    A_before = jnp.array(map_data_before["A_before"][0])
    c_before = jnp.array(map_data_before["c_before"][0])

    # Load and extract data (after training)
    BEST_NOMAP_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')["RMSE_after"][0]

    CONTR_after = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_after.npz')
    CONTR_after = mlp_controller.extract_params_from_batch(CONTR_after, 0) # controller data are always saved as batches
    controller_after = mlp_controller.update_params(CONTR_after)
    powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')
    BEST_NOMAP_powers_msv_after = powers_msv_after["powers_msv_after"][0]

    robot_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    L_after = jnp.array(robot_data_after["L_after"][0])
    D_after = jnp.array(robot_data_after["D_after"][0])
    r_after = jnp.array(robot_data_after["r_after"][0])
    rho_after = jnp.array(robot_data_after["rho_after"][0])
    E_after = jnp.array(robot_data_after["E_after"][0])
    G_after = jnp.array(robot_data_after["G_after"][0])
    robot_after = robot.update_params({"L": L_after, "D": jnp.diag(D_after), "r": r_after, "rho": rho_after, "E": E_after, "G": G_after})

    map_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    A_after = jnp.array(map_data_after["A_after"][0])
    c_after = jnp.array(map_data_after["c_after"][0])

    # Compute mapping effort (after training)
    BEST_NOMAP_mapping_effort_after = mean_Ek_ratio(
        robot_after, 
        RON_dataset, 
        partial(map, A=A_after, c=c_after)
    )
    BEST_NOMAP_condAinv = jnp.linalg.cond(jnp.linalg.inv(A_after))

    if do_nomap_case:
        # Simulation before training
        print('Simulating best case (before training)...')
        q0 = A_before @ y_RONsaved[0] + c_before
        qd0 = A_before @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_before)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_before.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_before = sim_out_pcs.t
        q_PCS_before, qd_PCS_before = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_before = sim_out_pcs.u
        y_hat_pcs_before = jnp.linalg.solve(A_before, (q_PCS_before - c_before).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_before = jnp.linalg.solve(A_before, qd_PCS_before.T).T            # yd_hat(t) = inv(A) * qd(t)

        # Simulation after training
        print('Simulating best case (after training)...')
        q0 = A_after @ y_RONsaved[0] + c_after
        qd0 = A_after @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_after)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_after.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_after = sim_out_pcs.t
        q_PCS_after, qd_PCS_after = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_after = sim_out_pcs.u
        y_hat_pcs_after = jnp.linalg.solve(A_after, (q_PCS_after - c_after).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_after = jnp.linalg.solve(A_after, qd_PCS_after.T).T           # yd_hat(t) = inv(A) * qd(t)

        # Compute simulation metrics
        BEST_NOMAP_simulationPower = compute_simulation_power(u_pcs_after, qd_PCS_after)
        BEST_NOMAP_simulationAccuracy = compute_simulation_rmse(timePCS_after, y_hat_pcs_after, time_RONsaved, y_RONsaved)
        BEST_NOMAP_simulationMapeffort = compute_simulation_Ek_ratio(robot_after, timePCS_after, q_PCS_after, qd_PCS_after, time_RONsaved, yd_RONsaved)

        # Show loss curve
        plt.figure()
        plt.plot(range(n_epochs), NOMAP_train_loss_ts, 'r', label='train loss')
        plt.plot(onp.arange(1,n_epochs+1), NOMAP_val_loss_ts, 'b', label='validation loss')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Result for best initial guess')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_loss', bbox_inches='tight')
        #plt.show()

        # Show and save animation before training 
        animate_robot_matplotlib(
            robot = robot_before,
            t_list = saveat,
            q_list = q_PCS_before,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_before.gif',
        )

        # Show animation after training
        animate_robot_matplotlib(
            robot = robot_after,
            t_list = saveat,
            q_list = q_PCS_after,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_after.gif',
        )

        # Plot robot strains and control torque before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_before, q_PCS_before[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_before, q_PCS_before[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_before, q_PCS_before[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_before, u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_before, u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_before, u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_before', bbox_inches='tight')
        #plt.show()

        # Plot robot strains and control torque after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_after, q_PCS_after[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_after, q_PCS_after[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_after, q_PCS_after[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_after, u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_after, u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_after, u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_after', bbox_inches='tight')
        #plt.show()

        # Plot actuation power before training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_before, qd_PCS_before[:,i] * u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_before, qd_PCS_before[:,i+1] * u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_before, qd_PCS_before[:,i+2] * u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_before', bbox_inches='tight')
        #plt.show()

        # Plot actuation power after training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_after, qd_PCS_after[:,i] * u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_after, qd_PCS_after[:,i+1] * u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_after, qd_PCS_after[:,i+2] * u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_after', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_before, y_hat_pcs_before[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_before', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_after, y_hat_pcs_after[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_after', bbox_inches='tight')
        #plt.show()

        # Plot phase planes before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_before[:, i], yd_hat_pcs_before[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_before', bbox_inches='tight')
        #plt.show()

        # Plot phase planes after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_after[:, i], yd_hat_pcs_after[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_after', bbox_inches='tight')
        #plt.show()

        # Save in a text file all the parameters before and after training
        with open(plots_folder/test_case/'best_result_parameters.txt', 'w') as file:
            file.write(f'----------BEFORE TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_before}\n')
            file.write(f'D = {D_before}\n')
            file.write(f'r = {r_before}\n')
            file.write(f'rho = {rho_before}\n')
            file.write(f'E = {E_before}\n')
            file.write(f'G = {G_before}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_before}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_before)}\n')
            file.write(f'c = {c_before}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(powers_msv_before)}\n')
            file.write(f'\n\n----------AFTER TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_after}\n')
            file.write(f'D = {D_after}\n')
            file.write(f'r = {r_after}\n')
            file.write(f'rho = {rho_after}\n')
            file.write(f'E = {E_after}\n')
            file.write(f'G = {G_after}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_after}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_after)}\n')
            file.write(f'c = {c_after}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(BEST_NOMAP_powers_msv_after)}\n')


#plt.show()
plt.close() # close figures to free memory
# =====================================================
# 2.2 Diagonal mapping case
# =====================================================
if do_diagmap_case or do_overall:
    print(f'\n--- DIAG MAP CASE ---')
    test_case = '2.2_diagMap'
    (plots_folder/test_case).mkdir(parents=True, exist_ok=True)

    ##### ALL SAMPLES #####
    prefix = 'SAMPLES_DIAGMAP'

    # Load and extract data
    all_loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    all_rmse_before = onp.load(data_folder/test_case/f'{prefix}_all_rmse_before.npz')
    all_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')
    all_robot_params_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    all_robot_params_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    all_map_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    all_powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')

    all_train_loss_ts = all_loss_curves["train_losses_ts"]
    all_val_loss_ts = all_loss_curves["val_losses_ts"]
    all_train_mse_ts = all_loss_curves["train_MSEs_ts"]
    all_rmse_before = all_rmse_before["RMSE_before"]
    SAMPLES_DIAGMAP_all_rmse_after = all_rmse_after["RMSE_after"]
    n_samples = all_rmse_before.shape[0]
    SAMPLES_DIAGMAP_all_powers_msv_after = all_powers_msv_after["powers_msv_after"]
    n_epochs_samples = all_train_mse_ts.shape[1]

    # Compute "mapping effort" for each sample (after training)
    print('Computing mapping effort (after training)')
    SAMPLES_DIAGMAP_mapping_effort_after = []
    for i in range(n_samples):
        robot_i = robot.update_params({
            "L": jnp.array(all_robot_params_after["L_after"][i]), 
            "D": jnp.diag(all_robot_params_after["D_after"][i]),
            "r": jnp.array(all_robot_params_after["r_after"][i]),
            "rho": jnp.array(all_robot_params_after["rho_after"][i]),
            "E": jnp.array(all_robot_params_after["E_after"][i]),
            "G": jnp.array(all_robot_params_after["G_after"][i]),
        })
        map_i = partial(map, A=jnp.array(all_map_after["A_after"][i]), c=jnp.array(all_map_after["c_after"][i]))
        mapping_effort_i = mean_Ek_ratio(robot_i, RON_dataset, map_i)
        SAMPLES_DIAGMAP_mapping_effort_after.append(mapping_effort_i)

    if do_diagmap_case:
        # Plot comparison of all samples (RMSE)
        colors = plt.cm.viridis(onp.linspace(0,1,n_samples))

        plt.figure()
        plt.scatter(onp.arange(n_samples)+1, all_rmse_before, marker='x', c=colors, label='test RMSE before')
        plt.scatter(onp.arange(n_samples)+1, SAMPLES_DIAGMAP_all_rmse_after, marker='o', c=colors, label='test RMSE after')
        plt.scatter(onp.arange(n_samples)+1, onp.sqrt(all_train_mse_ts[:,-1]), marker='+', c=colors, label='final train RMSE')
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('sample n.')
        plt.ylabel('RMSE')
        plt.title(f'Results for various initial guesses')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_comparison', bbox_inches='tight')
        #plt.show()

        # Plot comparison of all samples (loss curves)
        plt.figure()
        for i in range(n_samples):
            plt.plot(range(n_epochs_samples), all_train_loss_ts[i], color=colors[i], label=f'train losses' if i == 0 else "")
            plt.plot(onp.arange(1, n_epochs_samples + 1), all_val_loss_ts[i], '--', color=colors[i], label=f'validation losses' if i == 0 else "")
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Results for all samples')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_losses', bbox_inches='tight')
        #plt.show()

        # Save text file with all initial and final pcs parameters for the robot
        with open(plots_folder/test_case/'samples_pcs_params_comparison.txt', 'w') as file:
            file.write(f'PCS parameters before and after training for all samples:\n\n')
            for i in range(n_samples):
                file.write(f'L = {all_robot_params_before["L_before"][i]} --> {all_robot_params_after["L_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'D = {all_robot_params_before["D_before"][i]} --> {all_robot_params_after["D_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'r = {all_robot_params_before["r_before"][i]} --> {all_robot_params_after["r_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'rho = {all_robot_params_before["rho_before"][i]} --> {all_robot_params_after["rho_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'E = {all_robot_params_before["E_before"][i]} --> {all_robot_params_after["E_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'G = {all_robot_params_before["G_before"][i]} --> {all_robot_params_after["G_after"][i]}\n')

    ##### BEST RESULT #####
    prefix = 'BEST_DIAGMAP'

    # Load and extract data (training)
    loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    DIAGMAP_train_loss_ts = loss_curves["train_losses_ts"][0]
    DIAGMAP_val_loss_ts = loss_curves["val_losses_ts"][0]
    n_epochs = len(DIAGMAP_train_loss_ts)

    # Load and extract data (before training)
    CONTR_before = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_before.npz')
    CONTR_before = mlp_controller.extract_params_from_batch(CONTR_before, 0) # controller data are always saved as batches
    controller_before = mlp_controller.update_params(CONTR_before)
    powers_msv_before = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_before.npz')
    powers_msv_before = powers_msv_before["powers_msv_before"][0]

    robot_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    L_before = jnp.array(robot_data_before["L_before"][0])
    D_before = jnp.array(robot_data_before["D_before"][0])
    r_before = jnp.array(robot_data_before["r_before"][0])
    rho_before = jnp.array(robot_data_before["rho_before"][0])
    E_before = jnp.array(robot_data_before["E_before"][0])
    G_before = jnp.array(robot_data_before["G_before"][0])
    robot_before = robot.update_params({"L": L_before, "D": jnp.diag(D_before), "r": r_before, "rho": rho_before, "E": E_before, "G": G_before})

    map_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_map_before.npz')
    A_before = jnp.array(map_data_before["A_before"][0])
    c_before = jnp.array(map_data_before["c_before"][0])

    # Load and extract data (after training)
    BEST_DIAGMAP_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')["RMSE_after"][0]

    CONTR_after = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_after.npz')
    CONTR_after = mlp_controller.extract_params_from_batch(CONTR_after, 0) # controller data are always saved as batches
    controller_after = mlp_controller.update_params(CONTR_after)
    powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')
    BEST_DIAGMAP_powers_msv_after = powers_msv_after["powers_msv_after"][0]

    robot_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    L_after = jnp.array(robot_data_after["L_after"][0])
    D_after = jnp.array(robot_data_after["D_after"][0])
    r_after = jnp.array(robot_data_after["r_after"][0])
    rho_after = jnp.array(robot_data_after["rho_after"][0])
    E_after = jnp.array(robot_data_after["E_after"][0])
    G_after = jnp.array(robot_data_after["G_after"][0])
    robot_after = robot.update_params({"L": L_after, "D": jnp.diag(D_after), "r": r_after, "rho": rho_after, "E": E_after, "G": G_after})

    map_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    A_after = jnp.array(map_data_after["A_after"][0])
    c_after = jnp.array(map_data_after["c_after"][0])

    # Compute mapping effort (after training)
    BEST_DIAGMAP_mapping_effort_after = mean_Ek_ratio(
        robot_after, 
        RON_dataset, 
        partial(map, A=A_after, c=c_after)
    )
    BEST_DIAGMAP_condAinv = jnp.linalg.cond(jnp.linalg.inv(A_after))

    if do_diagmap_case:
        # Simulation before training
        print('Simulating best case (before training)...')
        q0 = A_before @ y_RONsaved[0] + c_before
        qd0 = A_before @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_before)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_before.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_before = sim_out_pcs.t
        q_PCS_before, qd_PCS_before = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_before = sim_out_pcs.u
        y_hat_pcs_before = jnp.linalg.solve(A_before, (q_PCS_before - c_before).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_before = jnp.linalg.solve(A_before, qd_PCS_before.T).T            # yd_hat(t) = inv(A) * qd(t)

        # Simulation after training
        print('Simulating best case (after training)...')
        q0 = A_after @ y_RONsaved[0] + c_after
        qd0 = A_after @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_after)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_after.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_after = sim_out_pcs.t
        q_PCS_after, qd_PCS_after = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_after = sim_out_pcs.u
        y_hat_pcs_after = jnp.linalg.solve(A_after, (q_PCS_after - c_after).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_after = jnp.linalg.solve(A_after, qd_PCS_after.T).T           # yd_hat(t) = inv(A) * qd(t)

        # Compute simulation metrics
        BEST_DIAGMAP_simulationPower = compute_simulation_power(u_pcs_after, qd_PCS_after)
        BEST_DIAGMAP_simulationAccuracy = compute_simulation_rmse(timePCS_after, y_hat_pcs_after, time_RONsaved, y_RONsaved)
        BEST_DIAGMAP_simulationMapeffort = compute_simulation_Ek_ratio(robot_after, timePCS_after, q_PCS_after, qd_PCS_after, time_RONsaved, yd_RONsaved)

        # Show loss curve
        plt.figure()
        plt.plot(range(n_epochs), DIAGMAP_train_loss_ts, 'r', label='train loss')
        plt.plot(onp.arange(1,n_epochs+1), DIAGMAP_val_loss_ts, 'b', label='validation loss')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Result for best initial guess')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_loss', bbox_inches='tight')
        #plt.show()

        # Show and save animation before training 
        animate_robot_matplotlib(
            robot = robot_before,
            t_list = saveat,
            q_list = q_PCS_before,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_before.gif',
        )

        # Show animation after training
        animate_robot_matplotlib(
            robot = robot_after,
            t_list = saveat,
            q_list = q_PCS_after,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_after.gif',
        )

        # Plot robot strains and control torque before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_before, q_PCS_before[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_before, q_PCS_before[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_before, q_PCS_before[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_before, u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_before, u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_before, u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_before', bbox_inches='tight')
        #plt.show()

        # Plot robot strains and control torque after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_after, q_PCS_after[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_after, q_PCS_after[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_after, q_PCS_after[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_after, u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_after, u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_after, u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_after', bbox_inches='tight')
        #plt.show()

        # Plot actuation power before training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_before, qd_PCS_before[:,i] * u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_before, qd_PCS_before[:,i+1] * u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_before, qd_PCS_before[:,i+2] * u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_before', bbox_inches='tight')
        #plt.show()

        # Plot actuation power after training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_after, qd_PCS_after[:,i] * u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_after, qd_PCS_after[:,i+1] * u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_after, qd_PCS_after[:,i+2] * u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_after', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_before, y_hat_pcs_before[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_before', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_after, y_hat_pcs_after[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_after', bbox_inches='tight')
        #plt.show()

        # Plot phase planes before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_before[:, i], yd_hat_pcs_before[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_before', bbox_inches='tight')
        #plt.show()

        # Plot phase planes after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_after[:, i], yd_hat_pcs_after[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_after', bbox_inches='tight')
        #plt.show()

        # Save in a text file all the parameters before and after training
        with open(plots_folder/test_case/'best_result_parameters.txt', 'w') as file:
            file.write(f'----------BEFORE TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_before}\n')
            file.write(f'D = {D_before}\n')
            file.write(f'r = {r_before}\n')
            file.write(f'rho = {rho_before}\n')
            file.write(f'E = {E_before}\n')
            file.write(f'G = {G_before}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_before}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_before)}\n')
            file.write(f'c = {c_before}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(powers_msv_before)}\n')
            file.write(f'\n\n----------AFTER TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_after}\n')
            file.write(f'D = {D_after}\n')
            file.write(f'r = {r_after}\n')
            file.write(f'rho = {rho_after}\n')
            file.write(f'E = {E_after}\n')
            file.write(f'G = {G_after}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_after}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_after)}\n')
            file.write(f'c = {c_after}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(BEST_DIAGMAP_powers_msv_after)}\n')


#plt.show()
plt.close() # close figures to free memory
# =====================================================
# 3.1 No MLP case
# =====================================================
if do_nomlp_case or do_overall:
    print(f'\n--- NO MLP CASE ---')
    test_case = '3.1_noMLP'
    (plots_folder/test_case).mkdir(parents=True, exist_ok=True)

    ##### ALL SAMPLES #####
    prefix = 'SAMPLES_NOMLP'

    # Load and extract data
    all_loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    all_rmse_before = onp.load(data_folder/test_case/f'{prefix}_all_rmse_before.npz')
    all_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')
    all_robot_params_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    all_robot_params_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    all_map_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    all_powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')

    all_train_loss_ts = all_loss_curves["train_losses_ts"]
    all_val_loss_ts = all_loss_curves["val_losses_ts"]
    all_train_mse_ts = all_loss_curves["train_MSEs_ts"]
    all_rmse_before = all_rmse_before["RMSE_before"]
    SAMPLES_NOMLP_all_rmse_after = all_rmse_after["RMSE_after"]
    n_samples = all_rmse_before.shape[0]
    SAMPLES_NOMLP_all_powers_msv_after = all_powers_msv_after["powers_msv_after"]
    n_epochs_samples = all_train_mse_ts.shape[1]

    # Compute "mapping effort" for each sample (after training)
    print('Computing mapping effort (after training)')
    SAMPLES_NOMLP_mapping_effort_after = []
    for i in range(n_samples):
        robot_i = robot.update_params({
            "L": jnp.array(all_robot_params_after["L_after"][i]), 
            "D": jnp.diag(all_robot_params_after["D_after"][i]),
            "r": jnp.array(all_robot_params_after["r_after"][i]),
            "rho": jnp.array(all_robot_params_after["rho_after"][i]),
            "E": jnp.array(all_robot_params_after["E_after"][i]),
            "G": jnp.array(all_robot_params_after["G_after"][i]),
        })
        map_i = partial(map, A=jnp.array(all_map_after["A_after"][i]), c=jnp.array(all_map_after["c_after"][i]))
        mapping_effort_i = mean_Ek_ratio(robot_i, RON_dataset, map_i)
        SAMPLES_NOMLP_mapping_effort_after.append(mapping_effort_i)

    if do_nomlp_case:
        # Plot comparison of all samples (RMSE)
        colors = plt.cm.viridis(onp.linspace(0,1,n_samples))

        plt.figure()
        plt.scatter(onp.arange(n_samples)+1, all_rmse_before, marker='x', c=colors, label='test RMSE before')
        plt.scatter(onp.arange(n_samples)+1, SAMPLES_NOMLP_all_rmse_after, marker='o', c=colors, label='test RMSE after')
        plt.scatter(onp.arange(n_samples)+1, onp.sqrt(all_train_mse_ts[:,-1]), marker='+', c=colors, label='final train RMSE')
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('sample n.')
        plt.ylabel('RMSE')
        plt.title(f'Results for various initial guesses')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_comparison', bbox_inches='tight')
        #plt.show()

        # Plot comparison of all samples (loss curves)
        plt.figure()
        for i in range(n_samples):
            plt.plot(range(n_epochs_samples), all_train_loss_ts[i], color=colors[i], label=f'train losses' if i == 0 else "")
            plt.plot(onp.arange(1, n_epochs_samples + 1), all_val_loss_ts[i], '--', color=colors[i], label=f'validation losses' if i == 0 else "")
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Results for all samples')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_losses', bbox_inches='tight')
        #plt.show()

        # Save text file with all initial and final pcs parameters for the robot
        with open(plots_folder/test_case/'samples_pcs_params_comparison.txt', 'w') as file:
            file.write(f'PCS parameters before and after training for all samples:\n\n')
            for i in range(n_samples):
                file.write(f'L = {all_robot_params_before["L_before"][i]} --> {all_robot_params_after["L_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'D = {all_robot_params_before["D_before"][i]} --> {all_robot_params_after["D_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'r = {all_robot_params_before["r_before"][i]} --> {all_robot_params_after["r_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'rho = {all_robot_params_before["rho_before"][i]} --> {all_robot_params_after["rho_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'E = {all_robot_params_before["E_before"][i]} --> {all_robot_params_after["E_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'G = {all_robot_params_before["G_before"][i]} --> {all_robot_params_after["G_after"][i]}\n')

    ##### BEST RESULT #####
    prefix = 'BEST_NOMLP'

    # Load and extract data (training)
    loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    NOMLP_train_loss_ts = loss_curves["train_losses_ts"][0]
    NOMLP_val_loss_ts = loss_curves["val_losses_ts"][0]
    n_epochs = len(NOMLP_train_loss_ts)

    # Load and extract data (before training)
    CONTR_before = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_before.npz')
    CONTR_before = mlp_controller.extract_params_from_batch(CONTR_before, 0) # controller data are always saved as batches
    controller_before = mlp_controller.update_params(CONTR_before)
    powers_msv_before = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_before.npz')
    powers_msv_before = powers_msv_before["powers_msv_before"][0]

    robot_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    L_before = jnp.array(robot_data_before["L_before"][0])
    D_before = jnp.array(robot_data_before["D_before"][0])
    r_before = jnp.array(robot_data_before["r_before"][0])
    rho_before = jnp.array(robot_data_before["rho_before"][0])
    E_before = jnp.array(robot_data_before["E_before"][0])
    G_before = jnp.array(robot_data_before["G_before"][0])
    robot_before = robot.update_params({"L": L_before, "D": jnp.diag(D_before), "r": r_before, "rho": rho_before, "E": E_before, "G": G_before})

    map_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_map_before.npz')
    A_before = jnp.array(map_data_before["A_before"][0])
    c_before = jnp.array(map_data_before["c_before"][0])

    # Load and extract data (after training)
    BEST_NOMLP_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')["RMSE_after"][0]

    CONTR_after = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_after.npz')
    CONTR_after = mlp_controller.extract_params_from_batch(CONTR_after, 0) # controller data are always saved as batches
    controller_after = mlp_controller.update_params(CONTR_after)
    powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')
    BEST_NOMLP_powers_msv_after = powers_msv_after["powers_msv_after"][0]

    robot_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    L_after = jnp.array(robot_data_after["L_after"][0])
    D_after = jnp.array(robot_data_after["D_after"][0])
    r_after = jnp.array(robot_data_after["r_after"][0])
    rho_after = jnp.array(robot_data_after["rho_after"][0])
    E_after = jnp.array(robot_data_after["E_after"][0])
    G_after = jnp.array(robot_data_after["G_after"][0])
    robot_after = robot.update_params({"L": L_after, "D": jnp.diag(D_after), "r": r_after, "rho": rho_after, "E": E_after, "G": G_after})

    map_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    A_after = jnp.array(map_data_after["A_after"][0])
    c_after = jnp.array(map_data_after["c_after"][0])

    # Compute mapping effort (after training)
    BEST_NOMLP_mapping_effort_after = mean_Ek_ratio(
        robot_after, 
        RON_dataset, 
        partial(map, A=A_after, c=c_after)
    )
    BEST_NOMLP_condAinv = jnp.linalg.cond(jnp.linalg.inv(A_after))

    if do_nomlp_case:
        # Simulation before training
        print('Simulating best case (before training)...')
        q0 = A_before @ y_RONsaved[0] + c_before
        qd0 = A_before @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_before)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_before.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_before = sim_out_pcs.t
        q_PCS_before, qd_PCS_before = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_before = sim_out_pcs.u
        y_hat_pcs_before = jnp.linalg.solve(A_before, (q_PCS_before - c_before).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_before = jnp.linalg.solve(A_before, qd_PCS_before.T).T            # yd_hat(t) = inv(A) * qd(t)

        # Simulation after training
        print('Simulating best case (after training)...')
        q0 = A_after @ y_RONsaved[0] + c_after
        qd0 = A_after @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_after)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_after.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_after = sim_out_pcs.t
        q_PCS_after, qd_PCS_after = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_after = sim_out_pcs.u
        y_hat_pcs_after = jnp.linalg.solve(A_after, (q_PCS_after - c_after).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_after = jnp.linalg.solve(A_after, qd_PCS_after.T).T           # yd_hat(t) = inv(A) * qd(t)

        # Compute simulation metrics
        BEST_NOMLP_simulationPower = compute_simulation_power(u_pcs_after, qd_PCS_after)
        BEST_NOMLP_simulationAccuracy = compute_simulation_rmse(timePCS_after, y_hat_pcs_after, time_RONsaved, y_RONsaved)
        BEST_NOMLP_simulationMapeffort = compute_simulation_Ek_ratio(robot_after, timePCS_after, q_PCS_after, qd_PCS_after, time_RONsaved, yd_RONsaved)

        # Show loss curve
        plt.figure()
        plt.plot(range(n_epochs), NOMLP_train_loss_ts, 'r', label='train loss')
        plt.plot(onp.arange(1,n_epochs+1), NOMLP_val_loss_ts, 'b', label='validation loss')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Result for best initial guess')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_loss', bbox_inches='tight')
        #plt.show()

        # Show and save animation before training 
        animate_robot_matplotlib(
            robot = robot_before,
            t_list = saveat,
            q_list = q_PCS_before,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_before.gif',
        )

        # Show animation after training
        animate_robot_matplotlib(
            robot = robot_after,
            t_list = saveat,
            q_list = q_PCS_after,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_after.gif',
        )

        # Plot robot strains and control torque before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_before, q_PCS_before[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_before, q_PCS_before[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_before, q_PCS_before[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_before, u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_before, u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_before, u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_before', bbox_inches='tight')
        #plt.show()

        # Plot robot strains and control torque after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_after, q_PCS_after[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_after, q_PCS_after[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_after, q_PCS_after[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_after, u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_after, u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_after, u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_after', bbox_inches='tight')
        #plt.show()

        # Plot actuation power before training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_before, qd_PCS_before[:,i] * u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_before, qd_PCS_before[:,i+1] * u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_before, qd_PCS_before[:,i+2] * u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_before', bbox_inches='tight')
        #plt.show()

        # Plot actuation power after training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_after, qd_PCS_after[:,i] * u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_after, qd_PCS_after[:,i+1] * u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_after, qd_PCS_after[:,i+2] * u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_after', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_before, y_hat_pcs_before[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_before', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_after, y_hat_pcs_after[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_after', bbox_inches='tight')
        #plt.show()

        # Plot phase planes before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_before[:, i], yd_hat_pcs_before[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_before', bbox_inches='tight')
        #plt.show()

        # Plot phase planes after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_after[:, i], yd_hat_pcs_after[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_after', bbox_inches='tight')
        #plt.show()

        # Save in a text file all the parameters before and after training
        with open(plots_folder/test_case/'best_result_parameters.txt', 'w') as file:
            file.write(f'----------BEFORE TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_before}\n')
            file.write(f'D = {D_before}\n')
            file.write(f'r = {r_before}\n')
            file.write(f'rho = {rho_before}\n')
            file.write(f'E = {E_before}\n')
            file.write(f'G = {G_before}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_before}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_before)}\n')
            file.write(f'c = {c_before}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(powers_msv_before)}\n')
            file.write(f'\n\n----------AFTER TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_after}\n')
            file.write(f'D = {D_after}\n')
            file.write(f'r = {r_after}\n')
            file.write(f'rho = {rho_after}\n')
            file.write(f'E = {E_after}\n')
            file.write(f'G = {G_after}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_after}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_after)}\n')
            file.write(f'c = {c_after}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(BEST_NOMLP_powers_msv_after)}\n')


#plt.show()
plt.close() # close figures to free memory
# =====================================================
# 3.2 Regularization MLP case
# =====================================================
if do_regulmlp_case or do_overall:
    print(f'\n--- REGULARIZATION MLP CASE ---')
    test_case = '3.2_regulMLP'
    (plots_folder/test_case).mkdir(parents=True, exist_ok=True)

    ##### ALL SAMPLES #####
    prefix = 'SAMPLES_REGULMLP'

    # Load and extract data
    all_loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    all_rmse_before = onp.load(data_folder/test_case/f'{prefix}_all_rmse_before.npz')
    all_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')
    all_robot_params_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    all_robot_params_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    all_map_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    all_powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')

    all_train_loss_ts = all_loss_curves["train_losses_ts"]
    all_val_loss_ts = all_loss_curves["val_losses_ts"]
    all_train_mse_ts = all_loss_curves["train_MSEs_ts"]
    all_val_mse_ts = all_loss_curves["val_MSEs_ts"]
    all_rmse_before = all_rmse_before["RMSE_before"]
    SAMPLES_REGULMLP_all_rmse_after = all_rmse_after["RMSE_after"]
    n_samples = all_rmse_before.shape[0]
    SAMPLES_REGULMLP_all_powers_msv_after = all_powers_msv_after["powers_msv_after"]
    n_epochs_samples = all_train_mse_ts.shape[1]

    # Compute "mapping effort" for each sample (after training)
    print('Computing mapping effort (after training)')
    SAMPLES_REGULMLP_mapping_effort_after = []
    for i in range(n_samples):
        robot_i = robot.update_params({
            "L": jnp.array(all_robot_params_after["L_after"][i]), 
            "D": jnp.diag(all_robot_params_after["D_after"][i]),
            "r": jnp.array(all_robot_params_after["r_after"][i]),
            "rho": jnp.array(all_robot_params_after["rho_after"][i]),
            "E": jnp.array(all_robot_params_after["E_after"][i]),
            "G": jnp.array(all_robot_params_after["G_after"][i]),
        })
        map_i = partial(map, A=jnp.array(all_map_after["A_after"][i]), c=jnp.array(all_map_after["c_after"][i]))
        mapping_effort_i = mean_Ek_ratio(robot_i, RON_dataset, map_i)
        SAMPLES_REGULMLP_mapping_effort_after.append(mapping_effort_i)

    if do_regulmlp_case:
        # Plot comparison of all samples (RMSE)
        colors = plt.cm.viridis(onp.linspace(0,1,n_samples))

        plt.figure()
        plt.scatter(onp.arange(n_samples)+1, all_rmse_before, marker='x', c=colors, label='test RMSE before')
        plt.scatter(onp.arange(n_samples)+1, SAMPLES_REGULMLP_all_rmse_after, marker='o', c=colors, label='test RMSE after')
        plt.scatter(onp.arange(n_samples)+1, onp.sqrt(all_train_mse_ts[:,-1]), marker='+', c=colors, label='final train RMSE') # mse, not loss!
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('sample n.')
        plt.ylabel('RMSE')
        plt.title(f'Results for various initial guesses')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_comparison', bbox_inches='tight')
        #plt.show()

        # Plot comparison of all samples (MSE curves)
        plt.figure()
        for i in range(n_samples):
            plt.plot(range(n_epochs_samples), all_train_mse_ts[i], color=colors[i], label=f'train MSEs' if i == 0 else "") # mse, not loss!
            plt.plot(onp.arange(1, n_epochs_samples + 1), all_val_mse_ts[i], '--', color=colors[i], label=f'validation MSEs' if i == 0 else "") # mse, not loss!
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Results for all samples')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_losses', bbox_inches='tight')
        #plt.show()

        # Save text file with all initial and final pcs parameters for the robot
        with open(plots_folder/test_case/'samples_pcs_params_comparison.txt', 'w') as file:
            file.write(f'PCS parameters before and after training for all samples:\n\n')
            for i in range(n_samples):
                file.write(f'L = {all_robot_params_before["L_before"][i]} --> {all_robot_params_after["L_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'D = {all_robot_params_before["D_before"][i]} --> {all_robot_params_after["D_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'r = {all_robot_params_before["r_before"][i]} --> {all_robot_params_after["r_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'rho = {all_robot_params_before["rho_before"][i]} --> {all_robot_params_after["rho_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'E = {all_robot_params_before["E_before"][i]} --> {all_robot_params_after["E_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'G = {all_robot_params_before["G_before"][i]} --> {all_robot_params_after["G_after"][i]}\n')

    ##### BEST RESULT #####
    prefix = 'BEST_REGULMLP'

    # Load and extract data (training)
    loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    REGULMLP_train_loss_ts = loss_curves["train_losses_ts"][0]
    REGULMLP_val_loss_ts = loss_curves["val_losses_ts"][0]
    REGULMLP_train_mse_ts = loss_curves["train_MSEs_ts"][0]
    REGULMLP_val_mse_ts = loss_curves["val_MSEs_ts"][0]
    n_epochs = len(REGULMLP_train_loss_ts)

    # Load and extract data (before training)
    CONTR_before = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_before.npz')
    CONTR_before = mlp_controller.extract_params_from_batch(CONTR_before, 0) # controller data are always saved as batches
    controller_before = mlp_controller.update_params(CONTR_before)
    powers_msv_before = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_before.npz')
    powers_msv_before = powers_msv_before["powers_msv_before"][0]

    robot_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    L_before = jnp.array(robot_data_before["L_before"][0])
    D_before = jnp.array(robot_data_before["D_before"][0])
    r_before = jnp.array(robot_data_before["r_before"][0])
    rho_before = jnp.array(robot_data_before["rho_before"][0])
    E_before = jnp.array(robot_data_before["E_before"][0])
    G_before = jnp.array(robot_data_before["G_before"][0])
    robot_before = robot.update_params({"L": L_before, "D": jnp.diag(D_before), "r": r_before, "rho": rho_before, "E": E_before, "G": G_before})

    map_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_map_before.npz')
    A_before = jnp.array(map_data_before["A_before"][0])
    c_before = jnp.array(map_data_before["c_before"][0])

    # Load and extract data (after training)
    BEST_REGULMLP_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')["RMSE_after"][0]

    CONTR_after = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_after.npz')
    CONTR_after = mlp_controller.extract_params_from_batch(CONTR_after, 0) # controller data are always saved as batches
    controller_after = mlp_controller.update_params(CONTR_after)
    powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')
    BEST_REGULMLP_powers_msv_after = powers_msv_after["powers_msv_after"][0]

    robot_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    L_after = jnp.array(robot_data_after["L_after"][0])
    D_after = jnp.array(robot_data_after["D_after"][0])
    r_after = jnp.array(robot_data_after["r_after"][0])
    rho_after = jnp.array(robot_data_after["rho_after"][0])
    E_after = jnp.array(robot_data_after["E_after"][0])
    G_after = jnp.array(robot_data_after["G_after"][0])
    robot_after = robot.update_params({"L": L_after, "D": jnp.diag(D_after), "r": r_after, "rho": rho_after, "E": E_after, "G": G_after})

    map_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    A_after = jnp.array(map_data_after["A_after"][0])
    c_after = jnp.array(map_data_after["c_after"][0])

    # Compute mapping effort (after training)
    BEST_REGULMLP_mapping_effort_after = mean_Ek_ratio(
        robot_after, 
        RON_dataset, 
        partial(map, A=A_after, c=c_after)
    )
    BEST_REGULMLP_condAinv = jnp.linalg.cond(jnp.linalg.inv(A_after))

    if do_regulmlp_case:
        # Simulation before training
        print('Simulating best case (before training)...')
        q0 = A_before @ y_RONsaved[0] + c_before
        qd0 = A_before @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_before)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_before.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_before = sim_out_pcs.t
        q_PCS_before, qd_PCS_before = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_before = sim_out_pcs.u
        y_hat_pcs_before = jnp.linalg.solve(A_before, (q_PCS_before - c_before).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_before = jnp.linalg.solve(A_before, qd_PCS_before.T).T            # yd_hat(t) = inv(A) * qd(t)

        # Simulation after training
        print('Simulating best case (after training)...')
        q0 = A_after @ y_RONsaved[0] + c_after
        qd0 = A_after @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_after)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_after.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_after = sim_out_pcs.t
        q_PCS_after, qd_PCS_after = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_after = sim_out_pcs.u
        y_hat_pcs_after = jnp.linalg.solve(A_after, (q_PCS_after - c_after).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_after = jnp.linalg.solve(A_after, qd_PCS_after.T).T           # yd_hat(t) = inv(A) * qd(t)

        # Compute simulation metrics
        BEST_REGULMLP_simulationPower = compute_simulation_power(u_pcs_after, qd_PCS_after)
        BEST_REGULMLP_simulationAccuracy = compute_simulation_rmse(timePCS_after, y_hat_pcs_after, time_RONsaved, y_RONsaved)
        BEST_REGULMLP_simulationMapeffort = compute_simulation_Ek_ratio(robot_after, timePCS_after, q_PCS_after, qd_PCS_after, time_RONsaved, yd_RONsaved)

        # Show loss curve
        plt.figure()
        plt.plot(range(n_epochs), REGULMLP_train_mse_ts, 'r', label='train MSE')
        plt.plot(onp.arange(1,n_epochs+1), REGULMLP_val_mse_ts, 'b', label='validation MSE')
        plt.plot(range(n_epochs), REGULMLP_train_loss_ts, 'r--', label='train loss')
        plt.plot(onp.arange(1,n_epochs+1), REGULMLP_val_loss_ts, 'b--', label='validation loss')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Result for best initial guess')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_loss', bbox_inches='tight')
        #plt.show()

        # Show and save animation before training 
        animate_robot_matplotlib(
            robot = robot_before,
            t_list = saveat,
            q_list = q_PCS_before,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_before.gif',
        )

        # Show animation after training
        animate_robot_matplotlib(
            robot = robot_after,
            t_list = saveat,
            q_list = q_PCS_after,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_after.gif',
        )

        # Plot robot strains and control torque before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_before, q_PCS_before[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_before, q_PCS_before[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_before, q_PCS_before[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_before, u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_before, u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_before, u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_before', bbox_inches='tight')
        #plt.show()

        # Plot robot strains and control torque after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_after, q_PCS_after[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_after, q_PCS_after[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_after, q_PCS_after[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_after, u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_after, u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_after, u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_after', bbox_inches='tight')
        #plt.show()

        # Plot actuation power before training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_before, qd_PCS_before[:,i] * u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_before, qd_PCS_before[:,i+1] * u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_before, qd_PCS_before[:,i+2] * u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_before', bbox_inches='tight')
        #plt.show()

        # Plot actuation power after training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_after, qd_PCS_after[:,i] * u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_after, qd_PCS_after[:,i+1] * u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_after, qd_PCS_after[:,i+2] * u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_after', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_before, y_hat_pcs_before[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_before', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_after, y_hat_pcs_after[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_after', bbox_inches='tight')
        #plt.show()

        # Plot phase planes before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_before[:, i], yd_hat_pcs_before[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_before', bbox_inches='tight')
        #plt.show()

        # Plot phase planes after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_after[:, i], yd_hat_pcs_after[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_after', bbox_inches='tight')
        #plt.show()

        # Save in a text file all the parameters before and after training
        with open(plots_folder/test_case/'best_result_parameters.txt', 'w') as file:
            file.write(f'----------BEFORE TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_before}\n')
            file.write(f'D = {D_before}\n')
            file.write(f'r = {r_before}\n')
            file.write(f'rho = {rho_before}\n')
            file.write(f'E = {E_before}\n')
            file.write(f'G = {G_before}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_before}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_before)}\n')
            file.write(f'c = {c_before}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(powers_msv_before)}\n')
            file.write(f'\n\n----------AFTER TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_after}\n')
            file.write(f'D = {D_after}\n')
            file.write(f'r = {r_after}\n')
            file.write(f'rho = {rho_after}\n')
            file.write(f'E = {E_after}\n')
            file.write(f'G = {G_after}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_after}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_after)}\n')
            file.write(f'c = {c_after}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(BEST_REGULMLP_powers_msv_after)}\n')


#plt.show()
plt.close() # close figures to free memory
# =====================================================
# 3.3 Regularization map case
# =====================================================
if do_regulmap_case or do_overall:
    print(f'\n--- REGULARIZATION MAP CASE ---')
    test_case = '3.3_regulMap'
    (plots_folder/test_case).mkdir(parents=True, exist_ok=True)

    ##### ALL SAMPLES #####
    prefix = 'SAMPLES_REGULMAP'

    # Load and extract data
    all_loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    all_rmse_before = onp.load(data_folder/test_case/f'{prefix}_all_rmse_before.npz')
    all_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')
    all_robot_params_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    all_robot_params_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    all_map_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    all_powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')

    all_train_loss_ts = all_loss_curves["train_losses_ts"]
    all_val_loss_ts = all_loss_curves["val_losses_ts"]
    all_train_mse_ts = all_loss_curves["train_MSEs_ts"]
    all_val_mse_ts = all_loss_curves["val_MSEs_ts"]
    all_rmse_before = all_rmse_before["RMSE_before"]
    SAMPLES_REGULMAP_all_rmse_after = all_rmse_after["RMSE_after"]
    n_samples = all_rmse_before.shape[0]
    SAMPLES_REGULMAP_all_powers_msv_after = all_powers_msv_after["powers_msv_after"]
    n_epochs_samples = all_train_mse_ts.shape[1]

    # Compute "mapping effort" for each sample (after training)
    print('Computing mapping effort (after training)')
    SAMPLES_REGULMAP_mapping_effort_after = []
    for i in range(n_samples):
        robot_i = robot.update_params({
            "L": jnp.array(all_robot_params_after["L_after"][i]), 
            "D": jnp.diag(all_robot_params_after["D_after"][i]),
            "r": jnp.array(all_robot_params_after["r_after"][i]),
            "rho": jnp.array(all_robot_params_after["rho_after"][i]),
            "E": jnp.array(all_robot_params_after["E_after"][i]),
            "G": jnp.array(all_robot_params_after["G_after"][i]),
        })
        map_i = partial(map, A=jnp.array(all_map_after["A_after"][i]), c=jnp.array(all_map_after["c_after"][i]))
        mapping_effort_i = mean_Ek_ratio(robot_i, RON_dataset, map_i)
        SAMPLES_REGULMAP_mapping_effort_after.append(mapping_effort_i)

    if do_regulmap_case:
        # Plot comparison of all samples (RMSE)
        colors = plt.cm.viridis(onp.linspace(0,1,n_samples))

        plt.figure()
        plt.scatter(onp.arange(n_samples)+1, all_rmse_before, marker='x', c=colors, label='test RMSE before')
        plt.scatter(onp.arange(n_samples)+1, SAMPLES_REGULMAP_all_rmse_after, marker='o', c=colors, label='test RMSE after')
        plt.scatter(onp.arange(n_samples)+1, onp.sqrt(all_train_mse_ts[:,-1]), marker='+', c=colors, label='final train RMSE') # mse, not loss!
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('sample n.')
        plt.ylabel('RMSE')
        plt.title(f'Results for various initial guesses')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_comparison', bbox_inches='tight')
        #plt.show()

        # Plot comparison of all samples (MSE curves)
        plt.figure()
        for i in range(n_samples):
            plt.plot(range(n_epochs_samples), all_train_mse_ts[i], color=colors[i], label=f'train MSEs' if i == 0 else "") # mse, not loss!
            plt.plot(onp.arange(1, n_epochs_samples + 1), all_val_mse_ts[i], '--', color=colors[i], label=f'validation MSEs' if i == 0 else "") # mse, not loss!
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Results for all samples')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_losses', bbox_inches='tight')
        #plt.show()

        # Save text file with all initial and final pcs parameters for the robot
        with open(plots_folder/test_case/'samples_pcs_params_comparison.txt', 'w') as file:
            file.write(f'PCS parameters before and after training for all samples:\n\n')
            for i in range(n_samples):
                file.write(f'L = {all_robot_params_before["L_before"][i]} --> {all_robot_params_after["L_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'D = {all_robot_params_before["D_before"][i]} --> {all_robot_params_after["D_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'r = {all_robot_params_before["r_before"][i]} --> {all_robot_params_after["r_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'rho = {all_robot_params_before["rho_before"][i]} --> {all_robot_params_after["rho_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'E = {all_robot_params_before["E_before"][i]} --> {all_robot_params_after["E_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'G = {all_robot_params_before["G_before"][i]} --> {all_robot_params_after["G_after"][i]}\n')

    ##### BEST RESULT #####
    prefix = 'BEST_REGULMAP'

    # Load and extract data (training)
    loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    REGULMAP_train_loss_ts = loss_curves["train_losses_ts"][0]
    REGULMAP_val_loss_ts = loss_curves["val_losses_ts"][0]
    REGULMAP_train_mse_ts = loss_curves["train_MSEs_ts"][0]
    REGULMAP_val_mse_ts = loss_curves["val_MSEs_ts"][0]
    n_epochs = len(REGULMAP_train_loss_ts)

    # Load and extract data (before training)
    CONTR_before = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_before.npz')
    CONTR_before = mlp_controller.extract_params_from_batch(CONTR_before, 0) # controller data are always saved as batches
    controller_before = mlp_controller.update_params(CONTR_before)
    powers_msv_before = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_before.npz')
    powers_msv_before = powers_msv_before["powers_msv_before"][0]

    robot_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    L_before = jnp.array(robot_data_before["L_before"][0])
    D_before = jnp.array(robot_data_before["D_before"][0])
    r_before = jnp.array(robot_data_before["r_before"][0])
    rho_before = jnp.array(robot_data_before["rho_before"][0])
    E_before = jnp.array(robot_data_before["E_before"][0])
    G_before = jnp.array(robot_data_before["G_before"][0])
    robot_before = robot.update_params({"L": L_before, "D": jnp.diag(D_before), "r": r_before, "rho": rho_before, "E": E_before, "G": G_before})

    map_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_map_before.npz')
    A_before = jnp.array(map_data_before["A_before"][0])
    c_before = jnp.array(map_data_before["c_before"][0])

    # Load and extract data (after training)
    BEST_REGULMAP_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')["RMSE_after"][0]

    CONTR_after = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_after.npz')
    CONTR_after = mlp_controller.extract_params_from_batch(CONTR_after, 0) # controller data are always saved as batches
    controller_after = mlp_controller.update_params(CONTR_after)
    powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')
    BEST_REGULMAP_powers_msv_after = powers_msv_after["powers_msv_after"][0]

    robot_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    L_after = jnp.array(robot_data_after["L_after"][0])
    D_after = jnp.array(robot_data_after["D_after"][0])
    r_after = jnp.array(robot_data_after["r_after"][0])
    rho_after = jnp.array(robot_data_after["rho_after"][0])
    E_after = jnp.array(robot_data_after["E_after"][0])
    G_after = jnp.array(robot_data_after["G_after"][0])
    robot_after = robot.update_params({"L": L_after, "D": jnp.diag(D_after), "r": r_after, "rho": rho_after, "E": E_after, "G": G_after})

    map_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    A_after = jnp.array(map_data_after["A_after"][0])
    c_after = jnp.array(map_data_after["c_after"][0])

    # Compute mapping effort (after training)
    BEST_REGULMAP_mapping_effort_after = mean_Ek_ratio(
        robot_after, 
        RON_dataset, 
        partial(map, A=A_after, c=c_after)
    )
    BEST_REGULMAP_condAinv = jnp.linalg.cond(jnp.linalg.inv(A_after))

    if do_regulmap_case:
        # Simulation before training
        print('Simulating best case (before training)...')
        q0 = A_before @ y_RONsaved[0] + c_before
        qd0 = A_before @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_before)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_before.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_before = sim_out_pcs.t
        q_PCS_before, qd_PCS_before = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_before = sim_out_pcs.u
        y_hat_pcs_before = jnp.linalg.solve(A_before, (q_PCS_before - c_before).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_before = jnp.linalg.solve(A_before, qd_PCS_before.T).T            # yd_hat(t) = inv(A) * qd(t)

        # Simulation after training
        print('Simulating best case (after training)...')
        q0 = A_after @ y_RONsaved[0] + c_after
        qd0 = A_after @ yd_RONsaved[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_after)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_after.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_after = sim_out_pcs.t
        q_PCS_after, qd_PCS_after = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_after = sim_out_pcs.u
        y_hat_pcs_after = jnp.linalg.solve(A_after, (q_PCS_after - c_after).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_after = jnp.linalg.solve(A_after, qd_PCS_after.T).T           # yd_hat(t) = inv(A) * qd(t)

        # Compute simulation metrics
        BEST_REGULMAP_simulationPower = compute_simulation_power(u_pcs_after, qd_PCS_after)
        BEST_REGULMAP_simulationAccuracy = compute_simulation_rmse(timePCS_after, y_hat_pcs_after, time_RONsaved, y_RONsaved)
        BEST_REGULMAP_simulationMapeffort = compute_simulation_Ek_ratio(robot_after, timePCS_after, q_PCS_after, qd_PCS_after, time_RONsaved, yd_RONsaved)

        # Show loss curve
        plt.figure()
        plt.plot(range(n_epochs), REGULMAP_train_mse_ts, 'r', label='train MSE')
        plt.plot(onp.arange(1,n_epochs+1), REGULMAP_val_mse_ts, 'b', label='validation MSE')
        plt.plot(range(n_epochs), REGULMAP_train_loss_ts, 'r--', label='train loss')
        plt.plot(onp.arange(1,n_epochs+1), REGULMAP_val_loss_ts, 'b--', label='validation loss')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Result for best initial guess')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_loss', bbox_inches='tight')
        #plt.show()

        # Show and save animation before training 
        animate_robot_matplotlib(
            robot = robot_before,
            t_list = saveat,
            q_list = q_PCS_before,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_before.gif',
        )

        # Show animation after training
        animate_robot_matplotlib(
            robot = robot_after,
            t_list = saveat,
            q_list = q_PCS_after,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_after.gif',
        )

        # Plot robot strains and control torque before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_before, q_PCS_before[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_before, q_PCS_before[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_before, q_PCS_before[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_before, u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_before, u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_before, u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_before', bbox_inches='tight')
        #plt.show()

        # Plot robot strains and control torque after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_after, q_PCS_after[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_after, q_PCS_after[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_after, q_PCS_after[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_after, u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_after, u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_after, u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_after', bbox_inches='tight')
        #plt.show()

        # Plot actuation power before training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_before, qd_PCS_before[:,i] * u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_before, qd_PCS_before[:,i+1] * u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_before, qd_PCS_before[:,i+2] * u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_before', bbox_inches='tight')
        #plt.show()

        # Plot actuation power after training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_after, qd_PCS_after[:,i] * u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_after, qd_PCS_after[:,i+1] * u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_after, qd_PCS_after[:,i+2] * u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_after', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_before, y_hat_pcs_before[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_before', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_after, y_hat_pcs_after[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_after', bbox_inches='tight')
        #plt.show()

        # Plot phase planes before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_before[:, i], yd_hat_pcs_before[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_before', bbox_inches='tight')
        #plt.show()

        # Plot phase planes after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_after[:, i], yd_hat_pcs_after[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_after', bbox_inches='tight')
        #plt.show()

        # Save in a text file all the parameters before and after training
        with open(plots_folder/test_case/'best_result_parameters.txt', 'w') as file:
            file.write(f'----------BEFORE TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_before}\n')
            file.write(f'D = {D_before}\n')
            file.write(f'r = {r_before}\n')
            file.write(f'rho = {rho_before}\n')
            file.write(f'E = {E_before}\n')
            file.write(f'G = {G_before}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_before}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_before)}\n')
            file.write(f'c = {c_before}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(powers_msv_before)}\n')
            file.write(f'\n\n----------AFTER TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_after}\n')
            file.write(f'D = {D_after}\n')
            file.write(f'r = {r_after}\n')
            file.write(f'rho = {rho_after}\n')
            file.write(f'E = {E_after}\n')
            file.write(f'G = {G_after}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_after}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_after)}\n')
            file.write(f'c = {c_after}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(BEST_REGULMAP_powers_msv_after)}\n')


#plt.show()
plt.close() # close figures to free memory
# =====================================================
# 4.1 Coupled case
# =====================================================
if do_coupled_case:
    print(f'\n--- COUPLED RON CASE ---')
    test_case = '4.1_coupled'
    (plots_folder/test_case).mkdir(parents=True, exist_ok=True)

    ##### ALL SAMPLES #####
    prefix = 'SAMPLES_COUP'

    # Load and extract data
    all_loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    all_rmse_before = onp.load(data_folder/test_case/f'{prefix}_all_rmse_before.npz')
    all_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')
    all_robot_params_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    all_robot_params_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')

    all_train_loss_ts = all_loss_curves["train_losses_ts"]
    all_val_loss_ts = all_loss_curves["val_losses_ts"]
    all_train_mse_ts = all_loss_curves["train_MSEs_ts"]
    all_rmse_before = all_rmse_before["RMSE_before"]
    all_rmse_after = all_rmse_after["RMSE_after"]
    n_samples = all_rmse_before.shape[0]
    n_epochs_samples = all_train_mse_ts.shape[1]

    if do_coupled_case:
        # Plot comparison of all samples (RMSE)
        colors = plt.cm.viridis(onp.linspace(0,1,n_samples))

        plt.figure()
        plt.scatter(onp.arange(n_samples)+1, all_rmse_before, marker='x', c=colors, label='test RMSE before')
        plt.scatter(onp.arange(n_samples)+1, all_rmse_after, marker='o', c=colors, label='test RMSE after')
        plt.scatter(onp.arange(n_samples)+1, onp.sqrt(all_train_mse_ts[:,-1]), marker='+', c=colors, label='final train RMSE')
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('sample n.')
        plt.ylabel('RMSE')
        plt.title(f'Results for various initial guesses')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_comparison', bbox_inches='tight')
        #plt.show()

        # Plot comparison of all samples (loss curves)
        plt.figure()
        for i in range(n_samples):
            plt.plot(range(n_epochs_samples), all_train_loss_ts[i], color=colors[i], label=f'train losses' if i == 0 else "")
            plt.plot(onp.arange(1, n_epochs_samples + 1), all_val_loss_ts[i], '--', color=colors[i], label=f'validation losses' if i == 0 else "")
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Results for all samples')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'samples_losses', bbox_inches='tight')
        #plt.show()

        # Save text file with all initial and final pcs parameters for the robot
        with open(plots_folder/test_case/'samples_pcs_params_comparison.txt', 'w') as file:
            file.write(f'PCS parameters before and after training for all samples:\n\n')
            for i in range(n_samples):
                file.write(f'L = {all_robot_params_before["L_before"][i]} --> {all_robot_params_after["L_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'D = {all_robot_params_before["D_before"][i]} --> {all_robot_params_after["D_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'r = {all_robot_params_before["r_before"][i]} --> {all_robot_params_after["r_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'rho = {all_robot_params_before["rho_before"][i]} --> {all_robot_params_after["rho_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'E = {all_robot_params_before["E_before"][i]} --> {all_robot_params_after["E_after"][i]}\n')
            for i in range(n_samples):
                if i == 0:
                    file.write(f'\n')
                file.write(f'G = {all_robot_params_before["G_before"][i]} --> {all_robot_params_after["G_after"][i]}\n')

    ##### BEST RESULT #####
    prefix = 'BEST_COUP'

    # Load and extract data (training)
    loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
    COUP_train_loss_ts = loss_curves["train_losses_ts"][0]
    COUP_val_loss_ts = loss_curves["val_losses_ts"][0]
    n_epochs = len(COUP_train_loss_ts)

    # Load and extract data (before training)
    CONTR_before = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_before.npz')
    CONTR_before = mlp_controller.extract_params_from_batch(CONTR_before, 0) # controller data are always saved as batches
    controller_before = mlp_controller.update_params(CONTR_before)
    powers_msv_before = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_before.npz')
    powers_msv_before = powers_msv_before["powers_msv_before"][0]

    robot_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_before.npz')
    L_before = jnp.array(robot_data_before["L_before"][0])
    D_before = jnp.array(robot_data_before["D_before"][0])
    r_before = jnp.array(robot_data_before["r_before"][0])
    rho_before = jnp.array(robot_data_before["rho_before"][0])
    E_before = jnp.array(robot_data_before["E_before"][0])
    G_before = jnp.array(robot_data_before["G_before"][0])
    robot_before = robot.update_params({"L": L_before, "D": jnp.diag(D_before), "r": r_before, "rho": rho_before, "E": E_before, "G": G_before})

    map_data_before = onp.load(data_folder/test_case/f'{prefix}_all_data_map_before.npz')
    A_before = jnp.array(map_data_before["A_before"][0])
    c_before = jnp.array(map_data_before["c_before"][0])

    # Load and extract data (after training)
    BEST_COUP_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')["RMSE_after"][0]

    CONTR_after = mlp_controller.load_params(data_folder/test_case/f'{prefix}_all_data_controller_after.npz')
    CONTR_after = mlp_controller.extract_params_from_batch(CONTR_after, 0) # controller data are always saved as batches
    controller_after = mlp_controller.update_params(CONTR_after)
    powers_msv_after = onp.load(data_folder/test_case/f'{prefix}_all_powers_msv_after.npz')
    powers_msv_after = powers_msv_after["powers_msv_after"][0]

    robot_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_robot_after.npz')
    L_after = jnp.array(robot_data_after["L_after"][0])
    D_after = jnp.array(robot_data_after["D_after"][0])
    r_after = jnp.array(robot_data_after["r_after"][0])
    rho_after = jnp.array(robot_data_after["rho_after"][0])
    E_after = jnp.array(robot_data_after["E_after"][0])
    G_after = jnp.array(robot_data_after["G_after"][0])
    robot_after = robot.update_params({"L": L_after, "D": jnp.diag(D_after), "r": r_after, "rho": rho_after, "E": E_after, "G": G_after})

    map_data_after = onp.load(data_folder/test_case/f'{prefix}_all_data_map_after.npz')
    A_after = jnp.array(map_data_after["A_after"][0])
    c_after = jnp.array(map_data_after["c_after"][0])

    if do_coupled_case:
        # Simulation before training
        print('Simulating best case (before training)...')
        q0 = A_before @ y_RONsaved_coupled[0] + c_before
        qd0 = A_before @ yd_RONsaved_coupled[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_before)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_before.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_before = sim_out_pcs.t
        q_PCS_before, qd_PCS_before = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_before = sim_out_pcs.u
        y_hat_pcs_before = jnp.linalg.solve(A_before, (q_PCS_before - c_before).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_before = jnp.linalg.solve(A_before, qd_PCS_before.T).T            # yd_hat(t) = inv(A) * qd(t)

        # Simulation after training
        print('Simulating best case (after training)...')
        q0 = A_after @ y_RONsaved_coupled[0] + c_after
        qd0 = A_after @ yd_RONsaved_coupled[0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

        tau_fb = jax.jit(partial(tau_law, controller=controller_after)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

        start = time.perf_counter()
        sim_out_pcs = robot_after.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_fb,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )
        end = time.perf_counter()
        print(f'Elapsed time: {end-start} s')

        timePCS_after = sim_out_pcs.t
        q_PCS_after, qd_PCS_after = jnp.split(sim_out_pcs.y, 2, axis=1)
        u_pcs_after = sim_out_pcs.u
        y_hat_pcs_after = jnp.linalg.solve(A_after, (q_PCS_after - c_after).T).T # y_hat(t) = inv(A) * ( q(t) - c )
        yd_hat_pcs_after = jnp.linalg.solve(A_after, qd_PCS_after.T).T            # yd_hat(t) = inv(A) * qd(t)

        # Show loss curve
        plt.figure()
        plt.plot(range(n_epochs), COUP_train_loss_ts, 'r', label='train loss')
        plt.plot(onp.arange(1,n_epochs+1), COUP_val_loss_ts, 'b', label='validation loss')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Result for best initial guess')
        plt.legend()
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_loss', bbox_inches='tight')
        #plt.show()

        # Show and save animation before training 
        animate_robot_matplotlib(
            robot = robot_before,
            t_list = saveat,
            q_list = q_PCS_before,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_before.gif',
        )

        # Show animation after training
        animate_robot_matplotlib(
            robot = robot_after,
            t_list = saveat,
            q_list = q_PCS_after,
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/test_case/'best_result_animation_after.gif',
        )

        # Plot robot strains and control torque before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_before, q_PCS_before[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_before, q_PCS_before[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_before, q_PCS_before[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_before, u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_before, u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_before, u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_before', bbox_inches='tight')
        #plt.show()

        # Plot robot strains and control torque after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i in range(n_pcs):
            axs[0,0].plot(timePCS_after, q_PCS_after[:,i], label=f'segment {i+1}')
            axs[0,0].grid(True)
            axs[0,0].set_xlabel('t [s]')
            axs[0,0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0,0].set_title('Bending strain')
            axs[0,0].legend()
            axs[1,0].plot(timePCS_after, q_PCS_after[:,i+1], label=f'segment {i+1}')
            axs[1,0].grid(True)
            axs[1,0].set_xlabel('t [s]')
            axs[1,0].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1,0].set_title('Axial strain')
            axs[1,0].legend()
            axs[2,0].plot(timePCS_after, q_PCS_after[:,i+2], label=f'segment {i+1}')
            axs[2,0].grid(True)
            axs[2,0].set_xlabel('t [s]')
            axs[2,0].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2,0].set_title('Shear strain')
            axs[2,0].legend()
        for i in range(n_pcs):
            axs[0,1].plot(timePCS_after, u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0,1].grid(True)
            axs[0,1].set_xlabel('t [s]')
            axs[0,1].set_ylabel(r"$u_\mathrm{be}$ [$N \cdot m^{2}$]")
            axs[0,1].set_title('Bending actuation')
            axs[0,1].legend()
            axs[1,1].plot(timePCS_after, u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1,1].grid(True)
            axs[1,1].set_xlabel('t [s]')
            axs[1,1].set_ylabel(r"$u_\mathrm{ax}$ [$N \cdot m$]")
            axs[1,1].set_title('Axial actuation')
            axs[1,1].legend()
            axs[2,1].plot(timePCS_after, u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2,1].grid(True)
            axs[2,1].set_xlabel('t [s]')
            axs[2,1].set_ylabel(r"$u_\mathrm{sh}$ [$N \cdot m$]")
            axs[2,1].set_title('Shear actuation')
            axs[2,1].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_strains_after', bbox_inches='tight')
        #plt.show()

        # Plot actuation power before training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_before, qd_PCS_before[:,i] * u_pcs_before[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_before, qd_PCS_before[:,i+1] * u_pcs_before[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_before, qd_PCS_before[:,i+2] * u_pcs_before[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_before', bbox_inches='tight')
        #plt.show()

        # Plot actuation power after training
        fig, axs = plt.subplots(3,1, figsize=(10,6))
        for i in range(n_pcs):
            axs[0].plot(timePCS_after, qd_PCS_after[:,i] * u_pcs_after[:,i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS_after, qd_PCS_after[:,i+1] * u_pcs_after[:,i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS_after, qd_PCS_after[:,i+2] * u_pcs_after[:,i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_power_after', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved_coupled, y_RONsaved_coupled[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_before, y_hat_pcs_before[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved_coupled[:,i])-1, onp.max(y_RONsaved_coupled[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_before', bbox_inches='tight')
        #plt.show()

        # Plot y(t) and y_hat(t) after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(time_RONsaved_coupled, y_RONsaved_coupled[:,i], 'b--', label=r'$y_{RON}(t)$')
            ax.plot(timePCS_after, y_hat_pcs_after[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
            ax.grid(True)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('y, q')
            ax.set_title(f'Component {i+1}')
            ax.set_ylim([onp.min(y_RONsaved_coupled[:,i])-1, onp.max(y_RONsaved_coupled[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_time_after', bbox_inches='tight')
        #plt.show()

        # Plot phase planes before training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved_coupled[:, i], yd_RONsaved_coupled[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_before[:, i], yd_hat_pcs_before[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved_coupled[:,i])-1, onp.max(y_RONsaved_coupled[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved_coupled[:,i])-1, onp.max(yd_RONsaved_coupled[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_before', bbox_inches='tight')
        #plt.show()

        # Plot phase planes after training
        fig, axs = plt.subplots(3,2, figsize=(12,9))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_RONsaved_coupled[:, i], yd_RONsaved_coupled[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            ax.plot(y_hat_pcs_after[:, i], yd_hat_pcs_after[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
            ax.grid(True)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$\dot{y}$')
            ax.set_title(f'Component {i+1}')
            ax.set_xlim([onp.min(y_RONsaved_coupled[:,i])-1, onp.max(y_RONsaved_coupled[:,i])+1])
            ax.set_ylim([onp.min(yd_RONsaved_coupled[:,i])-1, onp.max(yd_RONsaved_coupled[:,i])+1])
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_folder/test_case/'best_result_RONvsPCS_phaseplane_after', bbox_inches='tight')
        #plt.show()

        # Save in a text file all the parameters before and after training
        with open(plots_folder/test_case/'best_result_parameters.txt', 'w') as file:
            file.write(f'----------BEFORE TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_before}\n')
            file.write(f'D = {D_before}\n')
            file.write(f'r = {r_before}\n')
            file.write(f'rho = {rho_before}\n')
            file.write(f'E = {E_before}\n')
            file.write(f'G = {G_before}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_before}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_before)}\n')
            file.write(f'c = {c_before}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(powers_msv_before)}\n')
            file.write(f'\n\n----------AFTER TRAINING----------\n')
            file.write(f'PCS:\n')
            file.write(f'L = {L_after}\n')
            file.write(f'D = {D_after}\n')
            file.write(f'r = {r_after}\n')
            file.write(f'rho = {rho_after}\n')
            file.write(f'E = {E_after}\n')
            file.write(f'G = {G_after}\n')
            file.write(f'\nMAP:\n')
            file.write(f'A = {A_after}\n')
            file.write(f'A_inv = {onp.linalg.inv(A_after)}\n')
            file.write(f'c = {c_after}\n')
            file.write(f'\nCONTROLLER:\n')
            file.write(f'RMS power on the test set = {onp.sqrt(powers_msv_after)}\n')


# =====================================================
# Overall comparison
# =====================================================
if do_overall:
    ##### ALL SAMPLES #####
    # Plot accuracy vs control effort (on the test set)
    plt.figure()
    plt.scatter(SAMPLES_REF_all_rmse_after, onp.sqrt(SAMPLES_REF_all_powers_msv_after), color='k', label='reference')
    plt.scatter(SAMPLES_NOPCS_all_rmse_after, onp.sqrt(SAMPLES_NOPCS_all_powers_msv_after), color='r', label='no pcs')
    plt.scatter(SAMPLES_NOMAP_all_rmse_after, onp.sqrt(SAMPLES_NOMAP_all_powers_msv_after), color='g', label='no map')
    plt.scatter(SAMPLES_DIAGMAP_all_rmse_after, onp.sqrt(SAMPLES_DIAGMAP_all_powers_msv_after), color='c', label='diagonal map')
    plt.vlines(SAMPLES_NOMLP_all_rmse_after, -1*onp.ones_like(SAMPLES_NOMLP_all_rmse_after), plt.ylim()[1]*onp.ones_like(SAMPLES_NOMLP_all_rmse_after), color='m', alpha=0.2, label='no fb controller')
    plt.scatter(SAMPLES_REGULMLP_all_rmse_after, onp.sqrt(SAMPLES_REGULMLP_all_powers_msv_after), color='y', label='control penality')
    plt.scatter(SAMPLES_REGULMAP_all_rmse_after, onp.sqrt(SAMPLES_REGULMAP_all_powers_msv_after), color='b', label='map penality')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('RMS error on test set')
    plt.ylabel('RMS power on test set')
    plt.title('Control effort vs accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'Pareto_accuracy_vs_controleffort_samples', bbox_inches='tight')
    #plt.show()

    # Plot accuracy vs mapping "effort" (on the test set)
    plt.figure()
    plt.scatter(SAMPLES_REF_all_rmse_after, SAMPLES_REF_mapping_effort_after, color='k', label='reference')
    plt.scatter(SAMPLES_NOPCS_all_rmse_after, SAMPLES_NOPCS_mapping_effort_after, color='r', label='no pcs')
    plt.scatter(SAMPLES_NOMAP_all_rmse_after, SAMPLES_NOMAP_mapping_effort_after, color='g', label='no map')
    plt.scatter(SAMPLES_DIAGMAP_all_rmse_after, SAMPLES_DIAGMAP_mapping_effort_after, color='c', label='diagonal map')
    plt.scatter(SAMPLES_NOMLP_all_rmse_after, SAMPLES_NOMLP_mapping_effort_after, color='m', label='no fb controller')
    plt.scatter(SAMPLES_REGULMLP_all_rmse_after, SAMPLES_REGULMLP_mapping_effort_after, color='y', label='control penality')
    plt.scatter(SAMPLES_REGULMAP_all_rmse_after, SAMPLES_REGULMAP_mapping_effort_after, color='b', label='map penality')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('RMS error on test set')
    plt.ylabel(r'mean $E_{k}$ ratio on test set')
    plt.title('Mapping effort vs accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'Pareto_accuracy_vs_mappingeffort_samples', bbox_inches='tight')
    #plt.show()

    ##### BEST CASES #####
    # Plot all best losses together
    plt.figure()
    plt.plot(1+onp.arange(len(REF_val_loss_ts)), REF_val_loss_ts, color='k', label='reference')
    plt.plot(1+onp.arange(len(NOPCS_val_loss_ts)), NOPCS_val_loss_ts, color='r', label='no pcs')
    plt.plot(1+onp.arange(len(NOMAP_val_loss_ts)), NOMAP_val_loss_ts, color='g', label='no map')
    plt.plot(1+onp.arange(len(DIAGMAP_val_loss_ts)), DIAGMAP_val_loss_ts, color='c', label='diagonal map')
    plt.plot(1+onp.arange(len(NOMLP_val_loss_ts)), NOMLP_val_loss_ts, color='m', label='no fb controller')
    plt.plot(1+onp.arange(len(REGULMLP_val_mse_ts)), REGULMLP_val_mse_ts, color='y', label='control penality')
    plt.plot(1+onp.arange(len(REGULMAP_val_mse_ts)), REGULMAP_val_mse_ts, color='b', label='map penality')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title('Validation MSEs')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(plots_folder/'All_cases_validation_mse', bbox_inches='tight')
    #plt.show()

    # Plot accuracy vs control effort (on the test set and on a simulation)
    plt.figure()

    if do_ref_case and do_nopcs_case and do_nomap_case and do_diagmap_case and do_nomlp_case and do_regulmlp_case and do_regulmap_case:
        plt.scatter(BEST_REF_simulationAccuracy, BEST_REF_simulationPower, color='k', marker='+')
        plt.scatter(BEST_NOPCS_simulationAccuracy, BEST_NOPCS_simulationPower, color='r', marker='+')
        if BEST_NOMAP_simulationAccuracy < 1e10 and BEST_NOMAP_simulationPower < 1e10:
            plt.scatter(BEST_NOMAP_simulationAccuracy, BEST_NOMAP_simulationPower, color='g', marker='+')
        plt.scatter(BEST_DIAGMAP_simulationAccuracy, BEST_DIAGMAP_simulationPower, color='c', marker='+')
        y_axis_max = plt.ylim()[1]
        plt.vlines(BEST_NOMLP_simulationAccuracy, -1, y_axis_max, color='m', linestyle='--', alpha=1)
        plt.scatter(BEST_REGULMLP_simulationAccuracy, BEST_REGULMLP_simulationPower, color='y', marker='+')
        if BEST_REGULMAP_simulationAccuracy < 1e10 and BEST_REGULMAP_simulationPower < 1e10:
            plt.scatter(BEST_REGULMAP_simulationAccuracy, BEST_REGULMAP_simulationPower, color='b', marker='+')

    plt.scatter(BEST_REF_rmse_after, onp.sqrt(BEST_REF_powers_msv_after), color='k', label='reference')
    plt.scatter(BEST_NOPCS_rmse_after, onp.sqrt(BEST_NOPCS_powers_msv_after), color='r', label='no pcs')
    plt.scatter(BEST_NOMAP_rmse_after, onp.sqrt(BEST_NOMAP_powers_msv_after), color='g', label='no map')
    plt.scatter(BEST_DIAGMAP_rmse_after, onp.sqrt(BEST_DIAGMAP_powers_msv_after), color='c', label='diagonal map')
    plt.vlines(BEST_NOMLP_rmse_after, -1, y_axis_max, color='m', alpha=1, label='no fb controller')
    plt.scatter(BEST_REGULMLP_rmse_after, onp.sqrt(BEST_REGULMLP_powers_msv_after), color='y', label='control penality')
    plt.scatter(BEST_REGULMAP_rmse_after, onp.sqrt(BEST_REGULMAP_powers_msv_after), color='b', label='map penality')

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('RMS error')
    plt.ylabel('RMS power')
    plt.title(r'Control effort vs accuracy' f'\n' r'($\bullet$ = test set, + = simulation)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'Pareto_accuracy_vs_controleffort_best', bbox_inches='tight')
    #plt.show()

    # Plot accuracy vs mapping "effort" (on the test set and on a simulation)
    plt.figure()

    if do_ref_case and do_nopcs_case and do_nomap_case and do_diagmap_case and do_nomlp_case and do_regulmlp_case and do_regulmap_case:
        plt.scatter(BEST_REF_simulationAccuracy, BEST_REF_simulationMapeffort, color='k', marker='+')
        plt.scatter(BEST_NOPCS_simulationAccuracy, BEST_NOPCS_simulationMapeffort, color='r', marker='+')
        plt.scatter(BEST_NOMAP_simulationAccuracy, BEST_NOMAP_simulationMapeffort, color='g', marker='+')
        plt.scatter(BEST_DIAGMAP_simulationAccuracy, BEST_DIAGMAP_simulationMapeffort, color='c', marker='+')
        plt.scatter(BEST_NOMLP_simulationAccuracy, BEST_NOMLP_simulationMapeffort, color='m', marker='+')
        plt.scatter(BEST_REGULMLP_simulationAccuracy, BEST_REGULMLP_simulationMapeffort, color='y', marker='+')
        plt.scatter(BEST_REGULMAP_simulationAccuracy, BEST_REGULMAP_simulationMapeffort, color='b', marker='+')

    plt.scatter(BEST_REF_rmse_after, BEST_REF_mapping_effort_after, color='k', label=f'reference (cond(A)={BEST_REF_condAinv:.2f})')
    plt.scatter(BEST_NOPCS_rmse_after, BEST_NOPCS_mapping_effort_after, color='r', label=f'no pcs (cond(A)={BEST_NOPCS_condAinv:.2f})')
    plt.scatter(BEST_NOMAP_rmse_after, BEST_NOMAP_mapping_effort_after, color='g', label=f'no map (cond(A)={BEST_NOMAP_condAinv:.2f})')
    plt.scatter(BEST_DIAGMAP_rmse_after, BEST_DIAGMAP_mapping_effort_after, color='c', label=f'diagonal map (cond(A)={BEST_DIAGMAP_condAinv:.2f})')
    plt.scatter(BEST_NOMLP_rmse_after, BEST_NOMLP_mapping_effort_after, color='m', label=f'no fb controller (cond(A)={BEST_NOMLP_condAinv:.2f})')
    plt.scatter(BEST_REGULMLP_rmse_after, BEST_REGULMLP_mapping_effort_after, color='y', label=f'control penality (cond(A)={BEST_REGULMLP_condAinv:.2f})')
    plt.scatter(BEST_REGULMAP_rmse_after, BEST_REGULMAP_mapping_effort_after, color='b', label=f'map penality (cond(A)={BEST_REGULMAP_condAinv:.2f})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('RMS error')
    plt.ylabel(r'mean $E_{k}$ ratio')
    plt.title(r'Mapping effort vs accuracy' f'\n' r'($\bullet$ = test set, + = simulation)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'Pareto_accuracy_vs_mappingeffort_best', bbox_inches='tight')
    #plt.show()
    
plt.close()