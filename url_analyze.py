
import os
import sys
import argparse
import requests
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import re
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import ssl
import socket
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from datetime import datetime
import csv
from tqdm import tqdm

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: reportlab not installed. PDF generation disabled.")
    print("Install with: pip install reportlab")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebVulnerabilityPredictor:
    def __init__(self, model_path='deep_model.keras', tokenizer_path='tokenizer.json',
                 label_mapping_path='label_to_int.txt', output_dir='ml_analyze_out'):
        """
        Initialize the vulnerability predictor with trained model and preprocessing components
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_mapping_path = label_mapping_path
        self.output_dir = output_dir

        # Create output directory
        self.create_output_directory()

        # Load model and preprocessing components
        self.load_model_components()

        # Configure requests session with retries and timeouts
        self.session = self.setup_session()

        # Common vulnerability indicators (fixed regex patterns)
        self.vuln_patterns = {
            'sql_injection': [
                r'mysql_error', r'ora-\d{5}', r'microsoft ole db provider',
                r'unclosed quotation mark', r'syntax error.*query'
            ],
            'xss': [
                r'<script.*?>', r'javascript:', r'onerror=', r'onload=',
                r'alert\(', r'document\.cookie'
            ],
            'path_traversal': [
                r'\.\.//', r'\.\.\\\\', r'%2e%2e%2f', r'%2e%2e\\\\'
            ],
            'server_disclosure': [
                r'server:\s*apache/[\d.]+', r'server:\s*nginx/[\d.]+',
                r'server:\s*microsoft-iis/[\d.]+', r'x-powered-by:'
            ],
            'debug_info': [
                r'stack trace', r'debug mode', r'exception.*?at\s',
                r'warning:', r'notice:', r'fatal error'
            ]
        }

    def create_output_directory(self):
        """Create output directory structure"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'pdf'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'csv'), exist_ok=True)
            logger.info(f"Output directory created: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            sys.exit(1)

    def setup_session(self):
        """Setup requests session with proper configuration"""
        session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Headers to appear more like a regular browser
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        return session

    def load_model_components(self):
        """Load the trained model, tokenizer, and label mappings"""
        print("Loading model components...")

        with tqdm(total=3, desc="Loading Components", unit="component") as pbar:
            try:
                # Load model
                if os.path.exists(self.model_path):
                    self.model = tf.keras.models.load_model(self.model_path)
                    logger.info(f"Model loaded successfully from {self.model_path}")
                    pbar.set_description("Model loaded")
                    pbar.update(1)
                else:
                    logger.error(f"Model file not found: {self.model_path}")
                    sys.exit(1)

                # Load tokenizer
                if os.path.exists(self.tokenizer_path):
                    with open(self.tokenizer_path, 'r') as f:
                        tokenizer_json = f.read()
                        self.tokenizer = tokenizer_from_json(tokenizer_json)
                    logger.info("Tokenizer loaded successfully")
                    pbar.set_description("Tokenizer loaded")
                    pbar.update(1)
                else:
                    logger.error(f"Tokenizer file not found: {self.tokenizer_path}")
                    sys.exit(1)

                # Load label mappings
                if os.path.exists(self.label_mapping_path):
                    self.label_to_int = {}
                    self.int_to_label = {}
                    with open(self.label_mapping_path, 'r') as f:
                        for line in f:
                            label, idx = line.strip().split(':')
                            self.label_to_int[label] = int(idx)
                            self.int_to_label[int(idx)] = label
                    logger.info(f"Loaded {len(self.label_to_int)} label mappings")
                    pbar.set_description("Labels loaded")
                    pbar.update(1)
                else:
                    logger.error(f"Label mapping file not found: {self.label_mapping_path}")
                    sys.exit(1)

            except Exception as e:
                logger.error(f"Error loading model components: {e}")
                sys.exit(1)

    def extract_website_features(self, url, progress_callback=None):
        """
        Extract comprehensive features from a website that could indicate vulnerabilities
        """
        features = {
            'url_analysis': '',
            'headers': '',
            'content': '',
            'technologies': '',
            'forms': '',
            'links': '',
            'errors': '',
            'security_headers': ''
        }

        feature_steps = [
            "Parsing URL",
            "Fetching content",
            "Analyzing headers",
            "Extracting content",
            "Detecting technologies",
            "Analyzing forms",
            "Scanning links",
            "Checking error patterns"
        ]

        try:
            if progress_callback:
                progress_callback("Parsing URL")

            parsed_url = urlparse(url)
            features['url_analysis'] = f"domain {parsed_url.netloc} path {parsed_url.path} query {parsed_url.query}"

            if progress_callback:
                progress_callback("Fetching content")

            # Make request with timeout
            response = self.session.get(url, timeout=10, allow_redirects=True, verify=False)

            if progress_callback:
                progress_callback("Analyzing headers")

            # Header Analysis
            headers_text = []
            for key, value in response.headers.items():
                headers_text.append(f"{key.lower()} {value.lower()}")

                # Check for security headers
                if key.lower() in ['x-frame-options', 'x-xss-protection', 'x-content-type-options',
                                   'strict-transport-security', 'content-security-policy']:
                    features['security_headers'] += f"{key} {value} "

            features['headers'] = ' '.join(headers_text)

            if progress_callback:
                progress_callback("Extracting content")

            # Content Analysis
            content = response.text
            soup = BeautifulSoup(content, 'html.parser')

            # Extract visible text
            visible_text = soup.get_text(separator=' ', strip=True)
            features['content'] = visible_text[:2000]  # Limit content size

            if progress_callback:
                progress_callback("Detecting technologies")

            # Technology Detection
            tech_indicators = []

            # Check for common frameworks/technologies
            if 'wp-content' in content or 'wordpress' in content.lower():
                tech_indicators.append('wordpress')
            if 'drupal' in content.lower():
                tech_indicators.append('drupal')
            if 'joomla' in content.lower():
                tech_indicators.append('joomla')
            if 'react' in content.lower():
                tech_indicators.append('react')
            if 'angular' in content.lower():
                tech_indicators.append('angular')
            if 'vue' in content.lower():
                tech_indicators.append('vue')

            # Check server header
            server = response.headers.get('server', '').lower()
            if server:
                tech_indicators.append(f"server {server}")

            # Check X-Powered-By
            powered_by = response.headers.get('x-powered-by', '').lower()
            if powered_by:
                tech_indicators.append(f"powered-by {powered_by}")

            features['technologies'] = ' '.join(tech_indicators)

            if progress_callback:
                progress_callback("Analyzing forms")

            # Form Analysis
            forms = soup.find_all('form')
            form_analysis = []
            for form in forms:
                method = form.get('method', 'get').lower()
                action = form.get('action', '')
                form_analysis.append(f"form {method} {action}")

                # Check for input fields that might be vulnerable
                inputs = form.find_all(['input', 'textarea'])
                for inp in inputs:
                    inp_type = inp.get('type', 'text').lower()
                    inp_name = inp.get('name', '').lower()
                    form_analysis.append(f"input {inp_type} {inp_name}")

            features['forms'] = ' '.join(form_analysis)

            if progress_callback:
                progress_callback("Scanning links")

            # Link Analysis
            links = soup.find_all('a', href=True)
            link_analysis = []
            for link in links[:50]:  # Limit number of links
                href = link['href'].lower()
                if any(param in href for param in ['id=', 'user=', 'admin', 'login', 'upload']):
                    link_analysis.append(f"link {href}")

            features['links'] = ' '.join(link_analysis)

            if progress_callback:
                progress_callback("Checking error patterns")

            # Error Pattern Detection
            error_indicators = []
            content_lower = content.lower()

            for vuln_type, patterns in self.vuln_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content_lower):
                        error_indicators.append(f"{vuln_type} pattern detected")

            features['errors'] = ' '.join(error_indicators)

        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            features['errors'] = f"connection error {str(e)}"
        except Exception as e:
            logger.warning(f"Feature extraction error for {url}: {e}")
            features['errors'] = f"analysis error {str(e)}"

        return features

    def features_to_text(self, features):
        """Convert extracted features to text format similar to training data"""
        text_parts = []

        # Combine all features into a single text
        for key, value in features.items():
            if value:
                text_parts.append(f"{key} {value}")

        combined_text = ' '.join(text_parts)

        # Add some vulnerability-related keywords based on detected patterns
        vuln_keywords = []
        content_lower = combined_text.lower()

        if any(word in content_lower for word in ['sql', 'database', 'mysql', 'error']):
            vuln_keywords.append('database injection vulnerability')

        if any(word in content_lower for word in ['script', 'javascript', 'xss']):
            vuln_keywords.append('cross site scripting vulnerability')

        if any(word in content_lower for word in ['upload', 'file', 'path']):
            vuln_keywords.append('file inclusion vulnerability')

        if any(word in content_lower for word in ['admin', 'login', 'authentication']):
            vuln_keywords.append('authentication bypass vulnerability')

        if vuln_keywords:
            combined_text += ' ' + ' '.join(vuln_keywords)

        return combined_text

    def predict_vulnerability(self, url, max_len=300):
        """
        Predict vulnerability for a given URL with progress tracking
        """
        logger.info(f"Analyzing URL: {url}")

        # Progress tracking
        with tqdm(total=6, desc="Vulnerability Analysis", unit="step") as pbar:

            pbar.set_description("Extracting features")
            def update_progress(step):
                pbar.set_description(step)

            # Extract features from website
            features = self.extract_website_features(url, update_progress)
            pbar.update(1)

            pbar.set_description("Converting to text")
            # Convert features to text
            text = self.features_to_text(features)
            logger.info(f"Extracted text features: {text[:200]}...")
            pbar.update(1)

            pbar.set_description("Tokenizing text")
            # Preprocess text using the same tokenizer
            sequences = self.tokenizer.texts_to_sequences([text])
            X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
            pbar.update(1)

            pbar.set_description("Running ML prediction")
            # Make prediction
            predictions = self.model.predict(X, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            pbar.update(1)

            pbar.set_description("Processing results")
            # Get predicted label
            predicted_label = self.int_to_label.get(predicted_class_idx, "Unknown")

            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = []
            for idx in top_3_indices:
                label = self.int_to_label.get(idx, f"Class_{idx}")
                conf = float(predictions[0][idx])
                top_3_predictions.append((label, conf))
            pbar.update(1)

            pbar.set_description("Generating recommendations")
            result = {
                'url': url,
                'predicted_vulnerability': predicted_label,
                'confidence': confidence,
                'top_3_predictions': top_3_predictions,
                'extracted_features': features,
                'risk_level': self.assess_risk_level(confidence, predicted_label),
                'recommendations': self.generate_recommendations(features, predicted_label),
                'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            pbar.update(1)

            pbar.set_description("Analysis complete")

        return result

    def assess_risk_level(self, confidence, predicted_label):
        """Assess risk level based on confidence and predicted vulnerability type"""
        if confidence < 0.3:
            return "LOW"
        elif confidence < 0.6:
            return "MEDIUM"
        elif confidence < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"

    def generate_recommendations(self, features, predicted_vulnerability):
        """Generate security recommendations based on analysis"""
        recommendations = []

        # General recommendations
        if 'security_headers' not in features or not features['security_headers']:
            recommendations.append("Implement security headers (X-Frame-Options, CSP, HSTS)")

        if 'server' in features.get('headers', '').lower():
            recommendations.append("Consider hiding server version information")

        if 'form' in features.get('forms', '').lower():
            recommendations.append("Ensure all forms use HTTPS and proper validation")

        # Vulnerability-specific recommendations
        vuln_lower = predicted_vulnerability.lower()

        if 'injection' in vuln_lower or 'sql' in vuln_lower:
            recommendations.extend([
                "Use parameterized queries to prevent SQL injection",
                "Implement input validation and sanitization",
                "Use ORM frameworks with built-in protection"
            ])

        if 'xss' in vuln_lower or 'scripting' in vuln_lower:
            recommendations.extend([
                "Implement Content Security Policy (CSP)",
                "Sanitize all user inputs before output",
                "Use HTTPOnly and Secure flags for cookies"
            ])

        if 'authentication' in vuln_lower or 'access' in vuln_lower:
            recommendations.extend([
                "Implement multi-factor authentication",
                "Use strong password policies",
                "Implement proper session management"
            ])

        if 'buffer' in vuln_lower or 'overflow' in vuln_lower:
            recommendations.extend([
                "Implement proper input length validation",
                "Use safe string handling functions",
                "Enable compiler-based buffer overflow protections"
            ])

        return recommendations[:5]  # Limit to top 5 recommendations

    def save_csv_report(self, result):
        """Save result to CSV format"""
        print("\nSaving CSV report...")

        with tqdm(total=3, desc="Generating CSV", unit="file") as pbar:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain = urlparse(result['url']).netloc.replace('.', '_').replace(':', '_')
            csv_filename = os.path.join(self.output_dir, 'csv', f'{domain}_{timestamp}.csv')

            pbar.set_description("Creating main CSV")
            # Prepare data for CSV
            csv_data = {
                'URL': [result['url']],
                'Analysis_Timestamp': [result['analysis_timestamp']],
                'Risk_Level': [result['risk_level']],
                'Predicted_Vulnerability': [result['predicted_vulnerability']],
                'Confidence': [f"{result['confidence']:.4f}"],
                'Top_Prediction_1': [f"{result['top_3_predictions'][0][0]} ({result['top_3_predictions'][0][1]:.3f})"],
                'Top_Prediction_2': [f"{result['top_3_predictions'][1][0]} ({result['top_3_predictions'][1][1]:.3f})"],
                'Top_Prediction_3': [f"{result['top_3_predictions'][2][0]} ({result['top_3_predictions'][2][1]:.3f})"],
                'Security_Headers_Present': [bool(result['extracted_features'].get('security_headers', ''))],
                'Technologies_Detected': [result['extracted_features'].get('technologies', 'None')],
                'Forms_Found': [bool(result['extracted_features'].get('forms', ''))],
                'Recommendations_Count': [len(result['recommendations'])]
            }

            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filename, index=False)
            pbar.update(1)

            pbar.set_description("Creating recommendations CSV")
            # Also save detailed recommendations
            rec_filename = os.path.join(self.output_dir, 'csv', f'{domain}_{timestamp}_recommendations.csv')
            rec_data = {'Recommendation': result['recommendations']}
            rec_df = pd.DataFrame(rec_data)
            rec_df.to_csv(rec_filename, index=False)
            pbar.update(1)

            pbar.set_description("Creating features CSV")
            # Save detailed features
            features_filename = os.path.join(self.output_dir, 'csv', f'{domain}_{timestamp}_features.csv')
            features_data = []
            for key, value in result['extracted_features'].items():
                features_data.append({'Feature_Type': key, 'Content': value[:500] if value else 'None'})

            features_df = pd.DataFrame(features_data)
            features_df.to_csv(features_filename, index=False)
            pbar.update(1)

        logger.info(f"CSV reports saved:")
        logger.info(f"  Main report: {csv_filename}")
        logger.info(f"  Recommendations: {rec_filename}")
        logger.info(f"  Features: {features_filename}")

        return csv_filename, rec_filename, features_filename

    def save_pdf_report(self, result):
        """Save result to PDF format"""
        if not PDF_AVAILABLE:
            logger.warning("PDF generation not available. Install reportlab.")
            return None

        print("\nGenerating PDF report...")

        with tqdm(total=5, desc="Generating PDF", unit="section") as pbar:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain = urlparse(result['url']).netloc.replace('.', '_').replace(':', '_')
            pdf_filename = os.path.join(self.output_dir, 'pdf', f'{domain}_{timestamp}.pdf')

            doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            pbar.set_description("Creating title section")
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("Website Vulnerability Analysis Report", title_style))
            story.append(Spacer(1, 12))
            pbar.update(1)

            pbar.set_description("Creating analysis details")
            # Basic Information
            story.append(Paragraph("<b>Analysis Details</b>", styles['Heading2']))

            basic_info = [
                ['URL:', result['url']],
                ['Analysis Date:', result['analysis_timestamp']],
                ['Risk Level:', result['risk_level']],
                ['Predicted Vulnerability:', result['predicted_vulnerability']],
                ['Confidence Score:', f"{result['confidence']:.2%}"]
            ]

            basic_table = Table(basic_info, colWidths=[2*inch, 4*inch])
            basic_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(basic_table)
            story.append(Spacer(1, 20))
            pbar.update(1)

            pbar.set_description("Creating predictions table")
            # Top Predictions
            story.append(Paragraph("<b>Top 3 Vulnerability Predictions</b>", styles['Heading2']))

            pred_data = [['Rank', 'Vulnerability Type', 'Confidence']]
            for i, (label, conf) in enumerate(result['top_3_predictions'], 1):
                pred_data.append([str(i), label, f"{conf:.2%}"])

            pred_table = Table(pred_data, colWidths=[0.8*inch, 3.5*inch, 1.2*inch])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(pred_table)
            story.append(Spacer(1, 20))
            pbar.update(1)

            pbar.set_description("Creating recommendations")
            # Security Recommendations
            story.append(Paragraph("<b>Security Recommendations</b>", styles['Heading2']))

            for i, rec in enumerate(result['recommendations'], 1):
                story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                story.append(Spacer(1, 6))

            story.append(Spacer(1, 20))
            pbar.update(1)

            pbar.set_description("Creating technical details")
            # Technical Details
            story.append(Paragraph("<b>Technical Analysis Details</b>", styles['Heading2']))

            features = result['extracted_features']
            tech_info = [
                ['Technologies Detected:', features.get('technologies', 'None')[:100]],
                ['Security Headers:', 'Present' if features.get('security_headers') else 'Missing'],
                ['Forms Detected:', 'Yes' if features.get('forms') else 'No'],
                ['Suspicious Links:', 'Found' if features.get('links') else 'None'],
                ['Error Patterns:', features.get('errors', 'None')[:100]]
            ]

            tech_table = Table(tech_info, colWidths=[2*inch, 4*inch])
            tech_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))

            story.append(tech_table)

            # Risk Assessment Box
            story.append(Spacer(1, 20))
            risk_color = colors.green if result['risk_level'] == 'LOW' else \
                        colors.orange if result['risk_level'] == 'MEDIUM' else \
                        colors.red if result['risk_level'] == 'HIGH' else colors.darkred

            risk_style = ParagraphStyle(
                'RiskBox',
                parent=styles['Normal'],
                fontSize=12,
                textColor=risk_color,
                alignment=1,
                borderWidth=1,
                borderColor=risk_color
            )

            story.append(Paragraph(f"<b>RISK LEVEL: {result['risk_level']}</b>", risk_style))

            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph("Generated by ML Vulnerability Analysis System",
                                  ParagraphStyle('Footer', parent=styles['Normal'],
                                               fontSize=8, textColor=colors.grey, alignment=1)))
            pbar.update(1)

        doc.build(story)
        logger.info(f"PDF report saved: {pdf_filename}")

        return pdf_filename

def main():
    parser = argparse.ArgumentParser(description='Website Vulnerability Predictor with Progress Tracking and Report Generation')
    parser.add_argument('url', help='URL to analyze for vulnerabilities')
    parser.add_argument('--model', default='deep_vuln_model.keras', help='Path to trained model')
    parser.add_argument('--tokenizer', default='tokenizer.json', help='Path to tokenizer file')
    parser.add_argument('--labels', default='label_to_int.txt', help='Path to label mapping file')
    parser.add_argument('--output-dir', default='ml_analyze_out', help='Output directory for reports')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-pdf', action='store_true', help='Skip PDF generation')
    parser.add_argument('--no-csv', action='store_true', help='Skip CSV generation')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate URL
    if not args.url.startswith(('http://', 'https://')):
        args.url = 'https://' + args.url

    try:
        print(f"🔍 Starting vulnerability analysis for: {args.url}")
        print(f"📁 Reports will be saved to: {args.output_dir}")

        # Initialize predictor
        predictor = WebVulnerabilityPredictor(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            label_mapping_path=args.labels,
            output_dir=args.output_dir
        )

        # Make prediction with progress tracking
        result = predictor.predict_vulnerability(args.url)

        # Display results
        print("\n" + "="*60)
        print("WEBSITE VULNERABILITY ANALYSIS REPORT")
        print("="*60)
        print(f"URL: {result['url']}")
        print(f"Analysis Time: {result['analysis_timestamp']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Predicted Vulnerability: {result['predicted_vulnerability']}")
        print(f"Confidence: {result['confidence']:.2%}")

        print("\nTop 3 Predictions:")
        for i, (label, conf) in enumerate(result['top_3_predictions'], 1):
            print(f"  {i}. {label} ({conf:.2%})")

        print("\nSecurity Recommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")

        if args.verbose:
            print("\nExtracted Features:")
            for key, value in result['extracted_features'].items():
                if value:
                    print(f"  {key}: {value[:100]}...")

        print("\n" + "="*60)

        # Save reports with progress tracking
        saved_files = []

        if not args.no_csv:
            try:
                csv_files = predictor.save_csv_report(result)
                saved_files.extend(csv_files)
                print("✅ CSV reports generated successfully")
            except Exception as e:
                logger.error(f"Failed to save CSV report: {e}")

        if not args.no_pdf:
            try:
                pdf_file = predictor.save_pdf_report(result)
                if pdf_file:
                    saved_files.append(pdf_file)
                    print("✅ PDF report generated successfully")
                else:
                    print("⚠️  PDF generation skipped (reportlab not available)")
            except Exception as e:
                logger.error(f"Failed to save PDF report: {e}")

        if saved_files:
            print(f"\n📋 Analysis Complete! Reports saved:")
            for file in saved_files:
                print(f"   📄 {os.path.basename(file)}")
            print(f"\n📁 All reports saved in: {args.output_dir}")
        else:
            print("\n⚠️  No reports were generated")

    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
