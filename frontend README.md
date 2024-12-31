# Fake News Detector - React Frontend

This project provides a frontend interface for detecting fake news using the Fake News Detection API. The user can input a news headline, and the app will classify it as either "Real" or "Fake" along with a confidence score, an explanation of the prediction, and a SHAP (SHapley Additive exPlanations) plot.

## Features

- **Text Input**: Allows users to enter a news headline or article text.
- **Prediction**: Displays whether the input text is classified as "Real" or "Fake".
- **Confidence**: Shows the confidence score of the prediction.
- **Explanation**: Provides a textual explanation of the prediction using SHAP values.
- **SHAP Plot**: Displays a graphical SHAP plot showing token contributions to the model’s decision.

## Prerequisites

Before running the frontend, ensure you have the following:

1. **React**: Make sure you have Node.js and npm installed. If not, download and install them from the official website: [Node.js](https://nodejs.org/).
2. **Backend**: Ensure the Fake News Detection API (Flask app) is running on `http://localhost:5000`. This backend should be deployed as outlined in the [backend README](#).

## Getting Started

### 1. Clone the Repository

Clone this repository to your local machine:

git clone <repository_url>
cd <repository_folder/fake-news-detector>


### 2. Install Dependencies

Navigate to the project directory and install the required dependencies using npm:

npm install


### 3. Run the React App
Once the dependencies are installed, start the React development server with:

npm start

### 4. Interact with the App
Enter a news headline or article text into the input box.
Click on Analyze to get the prediction and explanation.
View the Prediction, Confidence, Explanation, and the SHAP Plot.