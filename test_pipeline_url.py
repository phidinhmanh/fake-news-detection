import sys
import logging
import json
import os
from dotenv import load_dotenv
from sequential_adversarial.pipeline import SequentialAdversarialPipeline

# Load API Key
load_dotenv()

# Configure logging so we can see the pipeline output in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_pipeline_with_url():
    # A sample news article (AP News)
    # url = "https://apnews.com/article/artificial-intelligence-openai-chatgpt-8d41fe030eb9dbb644131df33db13d7e"
    # Using a reliable news source that doesn't block bots aggressively
    url = "https://thehackernews.com/"

    print(f"Testing pipeline with URL: {url}")
    
    # Run the pipeline
    # We set mock=False to use the real LLM (Ollama which is running in the background)
    pipeline = SequentialAdversarialPipeline(mock=False)
    result = pipeline.run(url)
    
    with open("pipeline_test_results.txt", "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("--- PIPELINE EXECUTION RESULT ---\n")
        f.write("="*50 + "\n")
        f.write(f"Source: {result.source}\n")
        f.write(f"URL Extracted Text (first 200 chars): {result.raw_text[:200]}...\n")
        f.write(f"Overall Manipulation Score: {result.overall_manipulation_score}\n")
        
        if result.verity_report:
            f.write(f"\nCONCLUSION: {result.verity_report.conclusion}\n")
            f.write(f"CONFIDENCE: {result.verity_report.confidence}\n")
            f.write("\n--- Verity Report ---\n")
            f.write(result.verity_report.markdown_report + "\n")
        
        f.write("\n--- TF-IDF Baseline Comparison ---\n")
        if result.tfidf_comparison:
            f.write(f"TF-IDF Verdict: {result.tfidf_comparison.tfidf_label}\n")
            f.write(f"Agreement with LLM: {result.tfidf_comparison.agreement}\n")
            f.write(f"Notes: {result.tfidf_comparison.disagreement_notes}\n")

        f.write(f"\nVisual Flowchart saved at: {result.visual_flowchart_path}\n")
        f.write(f"Saved to DB with ID: {result.saved_id}\n")

    print("Pipeline finished successfully! Check pipeline_test_results.txt")


if __name__ == "__main__":
    test_pipeline_with_url()
