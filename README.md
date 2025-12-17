# From Ashes to Insights: Building a Cricket AI

A grounded AI chatbot for cricket analytics that prevents hallucinations by connecting LLMs to verified, context-adjusted player performance data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🏏 Overview

This capstone project tackles the hallucination problem in Large Language Models by grounding them with verified cricket statistics. Instead of allowing the LLM to access raw data (which can lead to made-up answers), it can only query pre-built, verified functions that return factual data.

### Key Features

- **Context-Aware Metrics**: Player performances adjusted for opponent strength (ELO ratings) and venue difficulty
- **Three Cricket Formats**: Support for Test, ODI, and T20 cricket with format-specific weightings
- **Grounded AI**: LLM cannot hallucinate - it only accesses pre-verified utility functions
- **Flexible Analytics**: Filter by format, era, country, venue, or player role
- **Ball-by-Ball Precision**: Built on granular match data from 2001 onwards

## 📊 Methodology

### 1. Data Foundation

**Phase I: Impact Calculations**
- Collected batting, bowling, and fielding statistics per match
- Normalized scores and combined into Match Impact Score
- Added Man of the Match bonuses

**Phase II: Team ELO Ratings**
- Implemented ELO system to measure team strength over time
- Used to adjust player performances based on opposition quality

**Phase III: Venue Factors**
- Calculated Venue Batting Factor (VBF) and Venue Bowling Factor (VBoF)
- Adjusted impact scores for venue difficulty

**Phase IV: Final Rankings**
- Combined all adjustment layers
- Produced venue and opponent-adjusted Match Impact Scores

### 2. Application Layer

Three utility functions limit AI access to verified data:

1. **`find_player()`** - Look up career stats and return match data
2. **`select_best_XI()`** - Select optimal playing XI based on scores
3. **match_scorecard()** - Generate full match summary

All functions support filtering by time period, country, format, etc.

### 3. Grounding the AI

```
User Query → LLM Interprets Intent → Function Call → LLM Formats Answer
```

The AI becomes a reasoning layer over verified data, eliminating hallucinations.

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cricket-ai.git
cd cricket-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OpenAI API key to .env
```

### Running the Application

```bash
# Run the Flask app
python src/app.py

# Access the web interface
# Navigate to http://localhost:5000
```

### Data Processing Pipeline

```bash
# Process raw match data (if you have raw JSON files)
jupyter notebook notebooks/impact_all_formats.ipynb

# Run all phases to generate aggregated statistics
# Outputs: aggregated_test.csv, aggregated_odi.csv, aggregated_t20.csv
```

## 📁 Project Structure

```
cricket-ai/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
├── .gitignore                        # Git ignore rules
│
├── data/
│   ├── aggregated_test.csv           # Processed Test match statistics
│   ├── aggregated_odi.csv            # Processed ODI statistics
│   ├── aggregated_t20.csv            # Processed T20 statistics
│   ├── test.csv                      # Match-level Test data
│   ├── odi.csv                       # Match-level ODI data
│   ├── t20.csv                       # Match-level T20 data
│   └── README.md                     # Data documentation
│
├── notebooks/
│   ├── impact_all_formats.ipynb      # Main data processing pipeline
│   └── role_classification.ipynb     # Player role classification
│
├── src/
│   ├── app.py                        # Flask application
│   ├── templates/
│   │   ├── index.html               # Main web interface
│   │   └── chatbot.html             # AI chatbot interface
│   └── static/                      # CSS, JS, images
│
└── docs/
    └── Capstone.pdf                  # Full project presentation
```

## 🎯 Usage Examples

### Finding Player Statistics

```python
from src.app import find_player

# Get all-format stats for a player
player_data = find_player(
    player_name="V Kohli",
    country='India',
    timeframe=(2010, 2020),
    formats=['Test', 'ODI', 'T20']
)

# Returns aggregated stats and top 10 match performances
print(player_data['Aggregated'])
print(player_data['Matches'])
```

### Selecting Best XI

```python
from src.app import select_best_XI

# Get best ODI XI for India
best_team = select_best_XI(
    file='data/aggregated_odi.csv',
    match_format='ODI',
    country='India',
    timeframe=(2010, 2020),
    num_roles={
        'Batsman': 5,
        'Allrounder': 2,
        'Wicketkeeper': 1,
        'Bowler': 3
    }
)
```

### Match Scorecards

```python
from src.app import match_scorecard

# Get matches between India and Australia in T20 World Cup 2024
matches = match_scorecard(
    match_format='T20',
    timeframe=('2024-06-01', '2024-06-30'),
    team1='India',
    team2='Australia'
)
```

## 🤖 AI Chatbot

The chatbot uses OpenAI's GPT-4 with function calling to access cricket data:

**Example Queries:**
- "Who is the best T20 batsman from India between 2015-2020?"
- "Compare Virat Kohli and Steve Smith in Test cricket"
- "Give me the best ODI XI made up of Australian players"
- "Show me the scorecard for the 2019 World Cup final"

The AI **cannot hallucinate** because it only accesses pre-verified functions that return factual data.

## 📈 Key Metrics

- **Impact Score**: Weighted combination of batting, bowling, and fielding contributions
- **Career Batting Average**: Total runs / dismissals across career
- **Career Strike Rate**: (Total runs / balls faced) × 100
- **Career Economy**: Runs conceded per over
- **Final Score**: Role-weighted combination optimized per format

### Format-Specific Weightings

**Test Cricket** (emphasis on consistency):
- Batting Average: 60%, Strike Rate: 40%
- Wickets: 60%, Economy: 40%

**ODI Cricket** (balanced):
- Batting Average: 50%, Strike Rate: 50%
- Wickets: 50%, Economy: 50%

**T20 Cricket** (emphasis on aggression):
- Batting Average: 40%, Strike Rate: 60%
- Wickets: 40%, Economy: 60%

## 🎓 Methodology Details

### Impact Score Calculation

```python
Impact Score = (Batting Component × W_BAT) + 
               (Bowling Component × W_BOWL) + 
               (Fielding Component × W_FIELD) + 
               (MoM Bonus)
```

### Venue Adjustment

```python
VBF = 1 + (1 - Venue_Avg_Runs / League_Avg_Runs) × 0.5
VBoF = 1 + (1 - Venue_Avg_Wickets / League_Avg_Wickets) × 0.5
```

### ELO Rating System

```python
New_Rating = Old_Rating + K × (Actual_Score - Expected_Score)
Expected_Score = 1 / (1 + 10^((Opponent_Rating - Team_Rating) / 400))
```

## 🔧 Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
FLASK_SECRET_KEY=your_secret_key_here
FLASK_ENV=development
```

### Customizing Weights

Modify format weights in `src/app.py`:

```python
def final_score(row, match_format):
    if match_format == 'T20':
        return 0.4*row['Impact_Score'] + \
               0.4*row['Career_Batting_Average'] + \
               0.2*row['Career_Strike_Rate']
    # ... customize for each format and role
```

## ⚠️ Limitations

1. **Historical Gaps**: Ball-by-ball data only available from 2001 onwards
2. **Quantifiable Only**: Leadership, captaincy, and clutch performance not captured
3. **Model Assumptions**: ELO ratings and venue factors use simplified rules
4. **Query Scope**: Chatbot limited to dataset and predefined functions

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Flask**: Web application framework
- **OpenAI GPT-4**: Language model for chatbot
- **NumPy**: Numerical computations
- **Jupyter**: Interactive data exploration

## Acknowledgments

- Ball-by-ball cricket data from [Cricsheet](https://cricsheet.org/)
- OpenAI for GPT-4 API
- Cricket community for inspiration and feedback
