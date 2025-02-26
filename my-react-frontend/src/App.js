import React from 'react';
import SentimentAnalyzer from './components/SentimentAnalyzer';
import './styles/App.css';

function App() {
    return (
        <div className="App">
            <h1>Sentiment Analysis</h1>
            <SentimentAnalyzer />
        </div>
    );
}

export default App;