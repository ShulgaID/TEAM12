import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from examples.bipedal_stairs.control.stair_controller import StairControllerFSM
from examples.bipedal_stairs.sim.environment import BipedalStairsEnv
from examples.bipedal_stairs.communication.layer import CommunicationLayer

#Main simulation runner coordinating Simulator <-> Communication <-> Controller
class SimulationRunner:
    #Initialize simulation runner. Args: render: Enable visualization; max_steps: Maximum simulation steps
    def __init__(self, render: bool = False, max_steps: int = 10000):
        self.render = render
        self.max_steps = max_steps

        # Initialize communication layer
        self.comm_layer = CommunicationLayer()

        # Initialize controller
        self.controller = StairControllerFSM(step_height=0.3, step_depth=0.4)

        # Initialize environment
        self.env = BipedalStairsEnv(
            communication_layer=self.comm_layer,
            render=render
        )

        # Statistics
        self.total_steps = 0
        self.max_height = 0.0
        self.max_distance = 0.0

    #Run the main simulation loop
    def run(self) -> None:
        print("="*70)
        print("SIMULATION: Bipedal Robot Stairs Climbing with FSM Controller")
        print("="*70)
        print(f"Max Steps: {self.max_steps}")
        print(f"Render: {self.render}")
        print("-"*70)

        # Reset environment and controller
        self.env.reset()
        self.controller.reset()

        print("\n[INIT] System initialized")
        print(f"[INIT] Simulator ready")
        print(f"[INIT] Controller FSM: {self.controller.get_state_name()}")
        print(f"[INIT] Communication layer active")

        print("\n" + "="*70)
        print("SIMULATION LOOP STARTING")
        print("="*70 + "\n")

        step = 0
        try:
            while step < self.max_steps:
                # Get current simulation time
                current_time = self.env.get_time()

                #Step 1: Simulator sends observation
                # This happens inside env.step()

                #Step 2: Controller receives observation
                observation = self.comm_layer.receive_observation()

                if observation is not None:
                    #Step 3: Controller updates FSM
                    control_output = self.controller.update(current_time, observation)

                    #Step 4: Controller sends control command
                    controller_state = self.controller.get_state_name()
                    self.comm_layer.send_control_command(
                        current_time,
                        control_output,
                        controller_state
                    )

                    #Step 5: Send state report
                    state_info = self._get_controller_state_info()
                    self.comm_layer.send_state_report(current_time, state_info)

                #Step 6: Simulator executes step
                new_observation, is_fallen = self.env.step()

                #Update statistics
                if new_observation is not None:
                    torso_pos = new_observation.get("torso_position", [0, 0, 0])
                    self.max_height = max(self.max_height, torso_pos[2])
                    self.max_distance = max(self.max_distance, torso_pos[0])

                #Print progress every 500 steps
                if (step + 1) % 500 == 0:
                    self._print_progress(step + 1, current_time, new_observation)

                #Check termination conditions
                if is_fallen:
                    print(f"\n[TERM] Robot fallen at step {step + 1}")
                    break

                step += 1
                self.total_steps += 1

        except KeyboardInterrupt:
            print("\n[TERM] Simulation interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Simulation error: {e}")
            self.comm_layer.send_error(self.env.get_time(), str(e))
            raise

        finally:
            self._print_summary()
            self.env.close()

    #Get controller state information
    def _get_controller_state_info(self) -> dict:
        ctrl_state = self.controller.get_state()
        return {
            "state": ctrl_state.current_state.value,
            "phase": float(ctrl_state.phase),
            "step_count": ctrl_state.step_count,
            "time_in_state": float(ctrl_state.time_in_state),
        }

    #Print simulation progress
    def _print_progress(self, step: int, current_time: float, observation: dict) -> None:
        if observation is None:
            return

        torso_pos = observation.get("torso_position", [0, 0, 0])
        controller_state = self.controller.get_state_name()
        phase = self.controller.controller_state.phase
        step_count = self.controller.controller_state.step_count

        print(
            f"[STEP {step:5d}] "
            f"Time: {current_time:7.3f}s | "
            f"State: {controller_state:12s} | "
            f"Phase: {phase:5.2f} | "
            f"Height: {torso_pos[2]:6.3f}m | "
            f"Distance: {torso_pos[0]:6.3f}m | "
            f"Steps: {step_count}"
        )

    #Print simulation summary
    def _print_summary(self) -> None:
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)
        print(f"Total Steps Executed: {self.total_steps}")
        print(f"Maximum Height Reached: {self.max_height:.3f}m")
        print(f"Maximum Distance Traveled: {self.max_distance:.3f}m")
        print(f"Final FSM State: {self.controller.get_state_name()}")
        print(f"Final Step Count: {self.controller.controller_state.step_count}")

        # Communication statistics
        comm_stats = self.comm_layer.get_stats()
        print(f"\nCommunication Statistics:")
        print(f"  Total Messages Processed: {comm_stats['buffer_stats']['total_messages_processed']}")
        print(f"  Remaining Incoming: {comm_stats['buffer_stats']['incoming_count']}")
        print(f"  Remaining Outgoing: {comm_stats['buffer_stats']['outgoing_count']}")
        print(f"  Message Log Size: {comm_stats['message_log_size']}")

        print("\n" + "="*70)

    #Print message log for debugging
    def get_message_log(self, last_n: int = 20) -> None:
        print("\n" + "="*70)
        print(f"COMMUNICATION LOG (Last {last_n} messages)")
        print("="*70)

        messages = self.comm_layer.get_message_log(last_n)
        for msg in messages:
            print(f"\n[{msg['message_type']:20s}] "
                  f"Time: {msg['timestamp']:.3f}s | "
                  f"Seq: {msg['sequence_id']:4d}")
            print(f"  From: {msg['sender']:12s} → To: {msg['receiver']:12s}")

#Main entry point
def main():
    parser = argparse.ArgumentParser(
        description="Bipedal robot stairs climbing simulator with FSM controller"
    )
    parser.add_argument("--render", action="store_true", help="Enable visualization")
    parser.add_argument(
        "--steps", type=int, default=10000, help="Maximum simulation steps"
    )
    parser.add_argument(
        "--log", action="store_true", help="Print communication log after simulation"
    )
    args = parser.parse_args()

    # Create and run simulation
    runner = SimulationRunner(render=args.render, max_steps=args.steps)
    runner.run()

    # Print message log if requested
    if args.log:
        runner.get_message_log(last_n=50)


if __name__ == "__main__":
    main()
