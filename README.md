
# WhatsApp Sentiment Analysis

This project performs sentiment analysis on WhatsApp chat data to determine the overall sentiment of the conversation. It utilizes the Natural Language Toolkit (NLTK) library and the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool to analyze the sentiment of individual messages.

## Features

- **WhatsApp Data Processing**: Load and preprocess WhatsApp chat data to extract messages for analysis.
- **Sentiment Analysis**: Analyze the sentiment of each message using NLTK's VADER sentiment analyzer.
- **Sentiment Category Determination**: Determine the overall sentiment category (positive, negative, or neutral) of the conversation based on sentiment scores.
- **Web Interface**: Provides a simple web interface for users to upload a WhatsApp dataset file and view the sentiment analysis results.

## Usage

1. Clone the repository:

   ```
   git clone https://github.com/your_username/whatsapp-sentiment-analysis.git
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the Flask application:

   ```
   python app.py
   ```

4. Access the web interface in your browser at [http://localhost:5000](http://localhost:5000) and upload a WhatsApp dataset file for sentiment analysis.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to customize this README message according to your project's specific details and requirements. Additionally, you may want to include sections such as Installation, Dependencies, License, etc., depending on the complexity and scope of your project.
