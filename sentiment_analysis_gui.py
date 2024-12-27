import tkinter as tk
from tkinter import Text, messagebox, filedialog
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk
import json

# Download the necessary NLTK data
nltk.download("vader_lexicon")

# Initialize sentiment analysis models
vader_analyzer = SentimentIntensityAnalyzer()
bert_analyzer = pipeline("sentiment-analysis")

# Save results to a file
def save_results():
    output_text = output_textbox.get("1.0", tk.END).strip()
    if not output_text:
        messagebox.showerror("Error", "No analysis results to save!")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if file_path:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(output_text)
        messagebox.showinfo("Success", "Results saved successfully!")

# Function to perform sentiment analysis
def analyze_sentiment():
    input_text = input_textbox.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showerror("Error", "Please enter a paragraph to analyze!")
        return

    # VADER Analysis
    vader_scores = vader_analyzer.polarity_scores(input_text)
    vader_result = max(vader_scores, key=vader_scores.get)

    # BERT Analysis
    bert_result = bert_analyzer(input_text[:512])  # BERT accepts max 512 characters

    # Display Results
    output_textbox.delete("1.0", tk.END)
    output_textbox.insert(tk.END, "=== Sentiment Analysis Results ===\n\n")
    output_textbox.insert(tk.END, f"VADER Result: {vader_result} (Score: {vader_scores[vader_result]:.2f})\n")
    output_textbox.insert(tk.END, f"BERT Result: {bert_result[0]['label']} (Score: {bert_result[0]['score']:.2f})\n\n")

    output_textbox.insert(tk.END, "=== VADER Scores ===\n")
    output_textbox.insert(tk.END, f"Positive: {vader_scores['pos']:.2f}\n")
    output_textbox.insert(tk.END, f"Neutral: {vader_scores['neu']:.2f}\n")
    output_textbox.insert(tk.END, f"Negative: {vader_scores['neg']:.2f}\n\n")

    # Research Gap and Metrics
    output_textbox.insert(tk.END, "=== Research Insights ===\n")
    output_textbox.insert(tk.END, "Identified Gap: Lack of comparative sentiment analysis using VADER and BERT for long text inputs.\n")
    output_textbox.insert(tk.END, "Evaluation Metrics: Sentiment polarity scores, prediction accuracy comparison.\n")

# GUI Setup
app = tk.Tk()
app.title("Sentiment Analysis Tool")
app.geometry("600x600")

# Input Section
tk.Label(app, text="Enter Paragraph:", font=("Arial", 12)).pack(pady=10)
input_textbox = Text(app, height=10, width=70)
input_textbox.pack()

# Analyze Button
analyze_button = tk.Button(app, text="Analyze Sentiment", command=analyze_sentiment, bg="blue", fg="white", font=("Arial", 12))
analyze_button.pack(pady=10)

# Save Button
save_button = tk.Button(app, text="Save Results", command=save_results, bg="green", fg="white", font=("Arial", 12))
save_button.pack(pady=5)

# Output Section
tk.Label(app, text="Sentiment Analysis Results:", font=("Arial", 12)).pack(pady=10)
output_textbox = Text(app, height=15, width=70, state="normal", bg="#f0f0f0")
output_textbox.pack()

# Run the GUI
app.mainloop()
