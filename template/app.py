
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def find_s_algorithm(data, target):
    # Initialize with the most specific hypothesis
    specific_hypothesis = [''] * (len(data[0]) - 1)  # Exclude the label column
    
    for instance in data:
        if instance[-1] == target:
            attributes = instance[:-1]
            if specific_hypothesis[0] == '':
                # If first positive example, initialize hypothesis
                specific_hypothesis = attributes
            else:
                for j in range(len(specific_hypothesis)):
                    if specific_hypothesis[j] != attributes[j]:
                        specific_hypothesis[j] = '?'
    
    return specific_hypothesis

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('data')
    target = request.json.get('target')
    specific_hypothesis = find_s_algorithm(data, target)
    return jsonify(specific_hypothesis=specific_hypothesis)

if __name__ == '__main__':
    app.run(debug=True)
