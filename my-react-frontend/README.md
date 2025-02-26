# My React Sentiment Analysis Frontend

This project is a React application that utilizes a sentiment analysis model to provide live sentiment feedback based on user input. The application communicates with a Flask backend to analyze the sentiment of text reviews.

## Project Structure

```
my-react-frontend
├── public
│   ├── index.html          # Main HTML file for the React application
├── src
│   ├── components
│   │   └── SentimentAnalyzer.js  # Component for sentiment analysis
│   ├── App.js              # Main App component
│   ├── index.js            # Entry point of the React application
│   └── styles
│       └── App.css         # CSS styles for the application
├── package.json             # npm configuration file
└── README.md                # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd my-react-frontend
   ```

2. **Install dependencies:**
   ```
   npm install
   ```

3. **Run the application:**
   ```
   npm start
   ```

   This will start the development server and open the application in your default web browser.

## Usage

- Enter a review in the input field provided by the SentimentAnalyzer component.
- The application will send the review to the sentiment analysis backend and display the predicted sentiment (Positive or Negative) in real-time.

## Dependencies

- React
- Axios (for making API requests)

## License

This project is licensed under the MIT License.