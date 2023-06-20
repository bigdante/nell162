from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

# Load the data from the clean.json file
data = json.load(open("clean_data.json"))

# Save the data to the clean.json file


# Route for rendering the show.html template
@app.route('/')
def show_page():
    return render_template('show.html')

# Route for handling the /show request
@app.route('/show', methods=['POST'])
def show_data():
    input_number = int(request.form['number'])
    if input_number >= 0 and input_number < len(data):
        response = {
            'data': data[input_number],
            'index': input_number
        }
        print(response)
    else:
        response = {
            'data': None,
            'index': -1
        }
    return jsonify(response)

# Route for handling the /save request
@app.route('/save', methods=['POST'])
def save_data_request():
    data = request.get_json()['data']
    print(data)
    with open('save.json', 'a') as file:
        file.write(json.dumps(data)+"\n")
    return jsonify({'message': 'Data saved successfully'})

if __name__ == '__main__':
    app.run(debug=True)
