import json

# Define a function to calculate the percentage of correct predictions
def calculate_accuracy(json_file_path):
    try:
        # Load the JSON data from the file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        total_matches = 0
        correct_predictions = 0

        # Iterate through each league and match in the JSON data
        for league, matches in data.items():
            for match in matches:
                total_matches += 1
                # Compare the actual result with the prediction
                if match.get('Result') == match.get('Prediction'):
                    correct_predictions += 1

        # Calculate the accuracy as a percentage
        if total_matches == 0:
            accuracy = 0
        else:
            accuracy = (correct_predictions / total_matches) * 100

        return accuracy

    except Exception as e:
        print(f"Error reading or processing the JSON file: {e}")
        return None

# Define the path to the JSON file
json_file_path = 'static/future_matches.json'

# Calculate and print the accuracy
accuracy = calculate_accuracy(json_file_path)
if accuracy is not None:
    print(f"The prediction accuracy is: {accuracy:.2f}%")
else:
    print("An error occurred while calculating the accuracy.")
