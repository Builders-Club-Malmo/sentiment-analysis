import React, { useState } from 'react';

const SentimentAnalyzer = () => {
    const [review, setReview] = useState('');
    const [sentiment, setSentiment] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleInputChange = (event) => {
        setReview(event.target.value);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        setLoading(true);
        setSentiment(null);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ review }),
            });
            const data = await response.json();
            setSentiment(data.sentiment);
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h1>Sentiment Analyzer</h1>
            <form onSubmit={handleSubmit}>
                <textarea
                    value={review}
                    onChange={handleInputChange}
                    placeholder="Type your review here..."
                    rows="4"
                    cols="50"
                />
                <br />
                <button type="submit" disabled={loading}>
                    {loading ? 'Analyzing...' : 'Analyze Sentiment'}
                </button>
            </form>
            {sentiment && <h2>Sentiment: {sentiment}</h2>}
        </div>
    );
};

export default SentimentAnalyzer;