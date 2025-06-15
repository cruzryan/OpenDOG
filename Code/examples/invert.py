import json

FILENAME = "./walk.json" # Use a constant for the filename

def modify_walk_data(data):
    """
    Modifies the walk data in place by inverting the sign of
    FL_tigh_actuator and FR_tigh_actuator values.
    """
    for entry in data:
        if "targets_deg" in entry:
            targets = entry["targets_deg"]
            if "FL_tigh_actuator" in targets:
                targets["FL_tigh_actuator"] *= -1
            if "FR_tigh_actuator" in targets:
                targets["FR_tigh_actuator"] *= -1
    return data # Return modified data (though it's modified in-place)

def main():
    try:
        # 1. Open and read the JSON file
        with open(FILENAME, 'r') as f:
            walk_data = json.load(f)
        print(f"Successfully loaded '{FILENAME}'.")

    except FileNotFoundError:
        print(f"Error: File '{FILENAME}' not found. Please create it with the sample content.")
        # For demonstration, let's create it if it doesn't exist
        print(f"Creating a sample '{FILENAME}' for demonstration...")
        sample_data = [
          {
            "duration": 0.2,
            "targets_deg": {
              "FR_tigh_actuator": 45.22,
              "FR_knee_actuator": 45.26,
              "FL_tigh_actuator": -44.78,
              "FL_knee_actuator": 39.06,
              "BR_tigh_actuator": 45.22,
              "BR_knee_actuator": -50.94,
              "BL_tigh_actuator": 45.22,
              "BL_knee_actuator": -44.74
            }
          },
          {
            "duration": 0.2,
            "targets_deg": {
              "FR_tigh_actuator": 45.22,
              "FR_knee_actuator": 42.57,
              "FL_tigh_actuator": -44.78,
              "FL_knee_actuator": 41.9,
              "BR_tigh_actuator": 45.22,
              "BR_knee_actuator": -48.1,
              "BL_tigh_actuator": 45.22,
              "BL_knee_actuator": -47.43
            }
          }
        ]
        with open(FILENAME, 'w') as f:
            json.dump(sample_data, f, indent=2)
        print(f"Sample '{FILENAME}' created. Please run the script again to modify it.")
        return # Exit after creating the sample
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{FILENAME}'. Please ensure it's valid JSON.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading: {e}")
        return

    # 2. Modify the data
    print("Original FL_tigh_actuator (first entry):", walk_data[0]["targets_deg"].get("FL_tigh_actuator"))
    print("Original FR_tigh_actuator (first entry):", walk_data[0]["targets_deg"].get("FR_tigh_actuator"))

    modified_data = modify_walk_data(walk_data) # Modifies walk_data in place

    print("Modified FL_tigh_actuator (first entry):", modified_data[0]["targets_deg"].get("FL_tigh_actuator"))
    print("Modified FR_tigh_actuator (first entry):", modified_data[0]["targets_deg"].get("FR_tigh_actuator"))

    try:
        # 3. Save the modified data back to the same file
        # The 'indent=2' argument makes the JSON file human-readable, similar to your example.
        # If your original file has a different indent (e.g., 4 spaces), adjust accordingly.
        with open(FILENAME, 'w') as f:
            json.dump(modified_data, f, indent=2)
        print(f"Successfully modified and saved data back to '{FILENAME}'.")

    except IOError:
        print(f"Error: Could not write to file '{FILENAME}'. Check permissions.")
    except Exception as e:
        print(f"An unexpected error occurred while writing: {e}")

if __name__ == "__main__":
    # Create a dummy walk.json if it doesn't exist for the script to run
    # You should have your actual walk.json file present.
    # This part is just for making the script runnable without manual file creation first.
    try:
        with open(FILENAME, 'r') as f:
            pass # File exists
    except FileNotFoundError:
        print(f"'{FILENAME}' not found. Creating a sample file for demonstration.")
        sample_data = [
          {
            "duration": 0.2,
            "targets_deg": {
              "FR_tigh_actuator": 45.22,
              "FR_knee_actuator": 45.26,
              "FL_tigh_actuator": -44.78,
              "FL_knee_actuator": 39.06,
              "BR_tigh_actuator": 45.22,
              "BR_knee_actuator": -50.94,
              "BL_tigh_actuator": 45.22,
              "BL_knee_actuator": -44.74
            }
          },
          {
            "duration": 0.2,
            "targets_deg": {
              "FR_tigh_actuator": 45.22,
              "FR_knee_actuator": 42.57,
              "FL_tigh_actuator": -44.78,
              "FL_knee_actuator": 41.9,
              "BR_tigh_actuator": 45.22,
              "BR_knee_actuator": -48.1,
              "BL_tigh_actuator": 45.22,
              "BL_knee_actuator": -47.43
            }
          }
        ]
        with open(FILENAME, 'w') as f_create:
            json.dump(sample_data, f_create, indent=2)
        print(f"Sample '{FILENAME}' created. You might want to run the script again.")

    main()