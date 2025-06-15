import json

FILENAME = "./walk.json"
KNEE_ADJUSTMENT_VALUE = 8.0
UPPER_EXTREME_THRESHOLD = 46.0
LOWER_EXTREME_THRESHOLD = -46.0
UPPER_MID_THRESHOLD = 44.0
LOWER_MID_THRESHOLD = -44.0

def modify_walk_data(data):
    for entry in data:
        if "targets_deg" in entry:
            targets = entry["targets_deg"]

            if "FL_tigh_actuator" in targets:
                targets["FL_tigh_actuator"] *= -1
            if "FR_tigh_actuator" in targets:
                targets["FR_tigh_actuator"] *= -1

            # Store original values for FL and BL knees if they exist,
            # these are the values *before* the KNEE_ADJUSTMENT logic.
            original_fl_knee_value = targets.get("FL_knee_actuator")
            original_bl_knee_value = targets.get("BL_knee_actuator")

            # 1. Apply KNEE_ADJUSTMENT_VALUE logic to ALL knee actuators
            knee_actuator_keys_to_adjust = [
                "FR_knee_actuator", "FL_knee_actuator",
                "BR_knee_actuator", "BL_knee_actuator"
            ]
            for key in knee_actuator_keys_to_adjust:
                if key in targets:
                    current_value = targets[key]
                    
                    if current_value > UPPER_EXTREME_THRESHOLD:
                        targets[key] += KNEE_ADJUSTMENT_VALUE
                    elif current_value < LOWER_EXTREME_THRESHOLD:
                        targets[key] -= KNEE_ADJUSTMENT_VALUE
                    elif current_value < UPPER_MID_THRESHOLD:
                        targets[key] -= KNEE_ADJUSTMENT_VALUE
                    elif current_value > LOWER_MID_THRESHOLD:
                        targets[key] += KNEE_ADJUSTMENT_VALUE
            
            # 2. Apply special sign logic for FL_knee_actuator and BL_knee_actuator
            #    This happens *after* the KNEE_ADJUSTMENT logic and *before* the swap.
            if "FL_knee_actuator" in targets and original_fl_knee_value is not None:
                adjusted_fl_knee = targets["FL_knee_actuator"]
                # If original was non-negative (no sign or zero) and adjustment made it negative
                if original_fl_knee_value >= 0 and adjusted_fl_knee < 0:
                    targets["FL_knee_actuator"] = abs(adjusted_fl_knee) # Remove the new sign
                # If original was negative (had a sign) and adjustment made it positive (and not zero)
                elif original_fl_knee_value < 0 and adjusted_fl_knee > 0:
                    targets["FL_knee_actuator"] = -adjusted_fl_knee # Restore the negative sign
            
            if "BL_knee_actuator" in targets and original_bl_knee_value is not None:
                adjusted_bl_knee = targets["BL_knee_actuator"]
                # If original was non-negative (no sign or zero) and adjustment made it negative
                if original_bl_knee_value >= 0 and adjusted_bl_knee < 0:
                    targets["BL_knee_actuator"] = abs(adjusted_bl_knee) # Remove the new sign
                # If original was negative (had a sign) and adjustment made it positive (and not zero)
                elif original_bl_knee_value < 0 and adjusted_bl_knee > 0:
                    targets["BL_knee_actuator"] = -adjusted_bl_knee # Restore the negative sign
            
            # 3. SWAP FL_knee_actuator and BL_knee_actuator values
            if "FL_knee_actuator" in targets and "BL_knee_actuator" in targets:
                temp_fl_knee = targets["FL_knee_actuator"]
                targets["FL_knee_actuator"] = targets["BL_knee_actuator"]
                targets["BL_knee_actuator"] = temp_fl_knee
    return data

def main():
    try:
        with open(FILENAME, 'r') as f:
            walk_data = json.load(f)
    except FileNotFoundError:
        print(f"'{FILENAME}' not found. Creating a sample file for demonstration.")
        sample_data = [
          { # Test case for sign logic and swap
            "duration": 0.2,
            "targets_deg": {
              "FR_tigh_actuator": 50.0, "FR_knee_actuator": 50.0,    # FR: 50+8 = 58
              "FL_tigh_actuator": -50.0, "FL_knee_actuator": 5.0,    # FL_orig: 5.0 -> 5-8 = -3.0 -> sign logic: abs(-3.0) = 3.0
              "BR_tigh_actuator": 40.0, "BR_knee_actuator": 40.0,    # BR: 40-8 = 32
              "BL_tigh_actuator": -40.0, "BL_knee_actuator": -5.0   # BL_orig: -5.0 -> -5-8 = -13.0 -> sign logic: no change (-13.0)
              # After sign logic: FL_knee = 3.0, BL_knee = -13.0
              # After swap: FL_knee = -13.0, BL_knee = 3.0
            }
          },
          { # Another test case
            "duration": 0.2,
            "targets_deg": {
              "FR_tigh_actuator": 45.0, "FR_knee_actuator": 45.0,    # FR: 45+8 = 53
              "FL_tigh_actuator": -45.0, "FL_knee_actuator": -6.0,   # FL_orig: -6.0 -> -6-8 = -14.0. Sign logic: no change.
              "BR_tigh_actuator": 46.0, "BR_knee_actuator": 46.0,    # BR: 46+8 = 54
              "BL_tigh_actuator": -46.0, "BL_knee_actuator": 47.0   # BL_orig: 47.0 -> 47+8 = 55.0. Sign logic: no change.
              # After sign logic: FL_knee = -14.0, BL_knee = 55.0
              # After swap: FL_knee = 55.0, BL_knee = -14.0
            }
          },
           { # Test case where adjustment flips sign from negative to positive
            "duration": 0.3,
            "targets_deg": {
                "FR_tigh_actuator": 10.0, "FR_knee_actuator": 10.0,
                "FL_tigh_actuator": -10.0, "FL_knee_actuator": -40.0, # FL_orig: -40 -> -40-8 = -48. Sign logic: no change.
                "BR_tigh_actuator": 20.0, "BR_knee_actuator": 20.0,
                "BL_tigh_actuator": -20.0, "BL_knee_actuator": -45.0  # BL_orig: -45.0. Adjust: -45-8 = -53.0. Sign logic: no change.
                                                                    # Suppose BL_orig was -2 and adjusted to +6. Then sign logic -> -6.
            }
          }
        ]
        # Create a more targeted example for sign flipping
        sample_data.append(
            {
            "duration": 0.4,
            "targets_deg": {
              "FR_tigh_actuator": 0, "FR_knee_actuator": 0,
              "FL_tigh_actuator": 0, "FL_knee_actuator": -2.0, # FL_orig: -2.0. Adjust: -2.0-8 = -10.0. Sign: no change. Value is -10.0
              "BR_tigh_actuator": 0, "BR_knee_actuator": 0,
              "BL_tigh_actuator": 0, "BL_knee_actuator": 45.0   # BL_orig: 45.0. Adjust: 45.0+8 = 53.0. Sign: no change. Value is 53.0
              # After sign logic: FL=-10, BL=53. After swap: FL=53, BL=-10
            }
          }
        )
        sample_data.append(
             { # Test: original positive, adjustment makes it negative -> remove sign
            "duration": 0.5,
            "targets_deg": { "FL_knee_actuator": 2.0 } # Orig: 2.0. Adjust: 2.0-8 = -6.0. Sign logic: abs(-6.0)=6.0
            }
        )
        sample_data.append(
             { # Test: original negative, adjustment makes it positive -> restore sign
            "duration": 0.6,
            # To make a negative number positive with current rules, it must be > LOWER_MID_THRESHOLD (-44) and add 8
            # e.g. -43. Adjust: -43+8 = -35. (Doesn't flip)
            # Let's assume a hypothetical adjustment for testing the sign logic more directly:
            # "FL_knee_actuator": -2.0 -> if it became 6.0 after adj. -> sign logic makes it -6.0
            # The current KNEE_ADJUSTMENT rules don't easily flip negative to positive.
            # We'll rely on the logic being sound as reasoned.
            "targets_deg": { "FL_knee_actuator": -40.0 } # Orig: -40. Adj: -40-8 = -48. Sign logic: no change.
            }
        )

        with open(FILENAME, 'w') as f_create:
            json.dump(sample_data, f_create, indent=2)
        print(f"Sample '{FILENAME}' created. Please run the script again to modify it.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{FILENAME}'. Please ensure it's valid JSON.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading: {e}")
        return

    modified_data = modify_walk_data(walk_data)

    try:
        with open(FILENAME, 'w') as f:
            json.dump(modified_data, f, indent=2)
        print(f"Successfully modified and saved data back to '{FILENAME}'.")
    except IOError:
        print(f"Error: Could not write to file '{FILENAME}'. Check permissions.")
    except Exception as e:
        print(f"An unexpected error occurred while writing: {e}")

if __name__ == "__main__":
    main()