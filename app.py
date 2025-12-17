from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from tabulate import tabulate
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Hardcoded CSV files for each format
csv_files = {
    "Test": "Data/aggregated_test.csv",
    "ODI": "Data/aggregated_odi.csv",
    "T20": "Data/aggregated_t20.csv"
}

# Additional mappings used by the `find_player` helper
combined_files = {
    "Test": "Data/aggregated_test.csv",
    "ODI": "Data/aggregated_odi.csv",
    "T20": "Data/aggregated_t20.csv"
}

match_files = {
    "Test": "Data/test.csv",
    "ODI": "Data/odi.csv",
    "T20": "Data/t20.csv"
}

def find_player(player_name, country='all', timeframe='all', formats=None):
    """Search aggregated stats and match-level files for a player.

    Returns a dict with keys 'Aggregated' and 'Matches' each containing a DataFrame.
    Handles name variations like "Joe Root" → "JE Root", "Virat Kohli" → "V Kohli", etc.
    """
    if formats is None:
        formats = ['Test', 'ODI', 'T20']

    player_records = []
    match_records = []

    # Function to handle name matching with variations
    def match_player_name(df, search_name):
        """Try to match player name with various formats"""
        search_lower = search_name.lower().strip()

        # Exact match
        exact_match = df[df['Player'].str.lower() == search_lower]
        if not exact_match.empty:
            return exact_match

        # Try partial match (contains)
        partial_match = df[df['Player'].str.lower().str.contains(search_lower, na=False)]
        if not partial_match.empty:
            return partial_match

        # Try matching by last name
        last_name = search_lower.split()[-1] if ' ' in search_lower else search_lower
        last_name_match = df[df['Player'].str.lower().str.contains(last_name, na=False)]
        if not last_name_match.empty:
            return last_name_match

        # Try matching with initials (e.g., "Joe Root" → "JE Root")
        parts = search_lower.split()
        if len(parts) >= 2:
            # Get first letter of first name + last name
            initial_last = parts[0][0] + ' ' + parts[-1]
            initial_match = df[df['Player'].str.lower().str.contains(initial_last, na=False)]
            if not initial_match.empty:
                return initial_match

        return pd.DataFrame()

    for fmt in formats:
        # --- Aggregated stats ---
        df_agg = pd.read_csv(combined_files.get(fmt))
        df_agg.replace([np.inf, -np.inf], 0, inplace=True)

        df_player_agg = match_player_name(df_agg, player_name).copy()

        if country != 'all':
            df_player_agg = df_player_agg[df_player_agg['Country'] == country].copy()
        if timeframe != 'all' and 'Year' in df_player_agg.columns:
            start_year, end_year = timeframe
            df_player_agg = df_player_agg[(df_player_agg['Year'] >= start_year) &
                                          (df_player_agg['Year'] <= end_year)].copy()
        if not df_player_agg.empty:
            # compute final score if final_score is available
            try:
                df_player_agg['Final_Score'] = df_player_agg.apply(lambda row: final_score(row, fmt), axis=1)
            except Exception:
                df_player_agg['Final_Score'] = df_player_agg.get('Impact_Score', 0)
            df_player_agg['Format'] = fmt
            player_records.append(df_player_agg)

        # --- Match performances ---
        df_match = pd.read_csv(match_files.get(fmt))
        df_match.replace([np.inf, -np.inf], 0, inplace=True)

        df_player_match = match_player_name(df_match, player_name).copy()

        if country != 'all' and 'Country' in df_player_match.columns:
            df_player_match = df_player_match[df_player_match['Country'] == country].copy()
        if timeframe != 'all' and 'Year' in df_player_match.columns:
            start_year, end_year = timeframe
            df_player_match = df_player_match[(df_player_match['Year'] >= start_year) &
                                              (df_player_match['Year'] <= end_year)].copy()
        if not df_player_match.empty:
            df_player_match['Format'] = fmt
            match_records.append(df_player_match)

    # Combine results
    results = {}
    results['Aggregated'] = pd.concat(player_records).reset_index(drop=True) if player_records else pd.DataFrame()
    if match_records:
        df_matches = pd.concat(match_records).reset_index(drop=True)
        if 'Impact_Score' in df_matches.columns:
            df_matches = df_matches.sort_values(by='Impact_Score', ascending=False).head(10)
        results['Matches'] = df_matches
    else:
        results['Matches'] = pd.DataFrame()

    return results

# Use your final_score function
def final_score(row, match_format):
    role = row['Predicted_Role']
    row['Career_Economy'] += 0.000001


    if match_format == 'T20':
        if role == 'Batsman':
            return 0.4*row['Impact_Score'] + 0.4*row['Career_Batting_Average'] + 0.2*row['Career_Strike_Rate']
        elif role == 'Bowler':
            return 0.4*row['Impact_Score'] + 0.4*row['Total_Wickets'] + 0.2*(1/row['Career_Economy'])
        elif role == 'Allrounder':
            batting = 0.4*row['Career_Batting_Average'] + 0.2*row['Career_Strike_Rate']
            bowling = 0.4*row['Total_Wickets'] + 0.2*(1/row['Career_Economy'])
            return 0.3*row['Impact_Score'] + 0.35*batting + 0.35*bowling
        elif role == 'Wicketkeeper':
            dismissals = row['Total_Fielding_Dismissals']
            return 0.3*row['Impact_Score'] + 0.5*row['Career_Batting_Average'] + 0.2*dismissals

    elif match_format == 'ODI':
        if role == 'Batsman':
            return 0.5*row['Impact_Score'] + 0.3*row['Career_Batting_Average'] + 0.2*row['Career_Strike_Rate']
        elif role == 'Bowler':
            return 0.5*row['Impact_Score'] + 0.3*row['Total_Wickets'] + 0.2*(1/row['Career_Economy'])
        elif role == 'Allrounder':
            batting = 0.5*row['Career_Batting_Average'] + 0.2*row['Career_Strike_Rate']
            bowling = 0.5*row['Total_Wickets'] + 0.2*(1/row['Career_Economy'])
            return 0.3*row['Impact_Score'] + 0.35*batting + 0.35*bowling
        elif role == 'Wicketkeeper':
            dismissals = row['Total_Fielding_Dismissals']
            return 0.4*row['Impact_Score'] + 0.4*row['Career_Batting_Average'] + 0.2*dismissals

    elif match_format == 'Test':
        if role == 'Batsman':
            return 0.6*row['Impact_Score'] + 0.4*row['Career_Batting_Average']
        elif role == 'Bowler':
            return 0.6*row['Impact_Score'] + 0.4*row['Total_Wickets']
        elif role == 'Allrounder':
            batting = 0.5*row['Career_Batting_Average']
            bowling = 0.5*row['Total_Wickets']
            return 0.3*row['Impact_Score'] + 0.35*batting + 0.35*bowling
        elif role == 'Wicketkeeper':
            dismissals = row['Total_Fielding_Dismissals']/row['Matches_Played']
            return 0.5*row['Impact_Score'] + 0.4*row['Career_Batting_Average'] + 0.1*dismissals

    else:
        return row['Impact_Score']  # fallback

# Use your select_best_XI function
def select_best_XI(file, match_format, country='all', timeframe='all', num_roles=None):
    if num_roles is None:
        num_roles = {'Batsman': 5, 'Allrounder': 1, 'Wicketkeeper': 1, 'Bowler': 4}

    df = pd.read_csv(file)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Filter by country
    if country != 'all':
        df = df[df['Country'] == country]

    # Filter by timeframe
    if timeframe != 'all' and 'Year' in df.columns:
        start_year, end_year = timeframe
        df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

    # Compute final scores
    df['Final_Score'] = df.apply(lambda row: final_score(row, match_format), axis=1)
    df = df.sort_values(by='Final_Score', ascending=False)

    # Select top players per role
    best_XI_list = []
    for role, n in num_roles.items():
        best_XI_list.append(df[df['Predicted_Role'] == role].head(n))

    best_XI = pd.concat(best_XI_list).reset_index(drop=True)
    return best_XI

def match_scorecard(match_format, timeframe='all', team1=None, team2=None, venue=None):
    """
    Returns a match scorecard for a given match format and timeframe,
    optionally filtering by two teams and/or venue.

    Parameters:
    - match_format: 'Test', 'ODI', or 'T20'
    - timeframe: 'all' or a tuple of ('YYYY-MM-DD', 'YYYY-MM-DD') for start and end date
    - team1: optional, first team
    - team2: optional, second team
    - venue: optional, venue name

    Returns:
    - pandas DataFrame sorted by Date, then Venue, then Team
    """
    # Load match data
    df_match = pd.read_csv(match_files.get(match_format))
    df_match.replace([np.inf, -np.inf], 0, inplace=True)
    df_match['Date'] = pd.to_datetime(df_match['Date'])

    # Filter by timeframe
    if timeframe != 'all':
        start_date, end_date = timeframe
        df_scorecard = df_match[(df_match['Date'] >= pd.to_datetime(start_date)) &
                                (df_match['Date'] <= pd.to_datetime(end_date))].copy()
    else:
        df_scorecard = df_match.copy()

    # Filter matches where both teams are present
    if team1 and team2:
        df_scorecard = df_scorecard[((df_scorecard['Team'] == team1) & (df_scorecard['Opponent'] == team2)) |
                                    ((df_scorecard['Team'] == team2) & (df_scorecard['Opponent'] == team1))].copy()
    elif team1:
        df_scorecard = df_scorecard[df_scorecard['Team'] == team1].copy()
    elif team2:
        df_scorecard = df_scorecard[df_scorecard['Team'] == team2].copy()

    # Filter by venue
    if venue:
        df_scorecard = df_scorecard[df_scorecard['Venue'].str.lower() == venue.lower()].copy()

    if df_scorecard.empty:
        print(f"No matches found for {match_format} with the specified filters")
        return pd.DataFrame()

    # Sort by Date, Venue, Team
    df_scorecard = df_scorecard.sort_values(by=['Date', 'Venue', 'Team'], ascending=[True, True, True])

    # Key columns for concise scorecard
    cols_to_show = [
        'Player', 'Date', 'Team', 'Opponent', 'Venue', 'MoM Awarded',
        'Runs', 'Balls Faced', 'Dismissals', 'Batting Average (Match)',
        'Strike Rate', 'Wickets', 'Balls Bowled', 'Economy',
        'Fielding Dismissals', 'Weighted Impact Score'
    ]
    df_scorecard = df_scorecard[[c for c in cols_to_show if c in df_scorecard.columns]]

    return df_scorecard.reset_index(drop=True)


# OpenAI Function Definitions for tool calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_player",
            "description": "Search for a cricket player's aggregated stats and match performances across multiple formats",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                        "description": "Name of the cricket player to search for"
                    },
                    "country": {
                        "type": "string",
                        "description": "Filter by country (e.g., 'India', 'Australia'). Use 'all' for no filter.",
                        "default": "all"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Filter by years in format 'YYYY-YYYY'. Use 'all' for no filter.",
                        "default": "all"
                    },
                    "formats": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["Test", "ODI", "T20"]},
                        "description": "Cricket formats to search (Test, ODI, T20)",
                        "default": ["Test", "ODI", "T20"]
                    }
                },
                "required": ["player_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_best_XI",
            "description": "Build a best XI team based on player performance scores and role",
            "parameters": {
                "type": "object",
                "properties": {
                    "match_format": {
                        "type": "string",
                        "enum": ["Test", "ODI", "T20"],
                        "description": "Cricket format (Test, ODI, or T20)"
                    },
                    "country": {
                        "type": "string",
                        "description": "Filter by country. Use 'all' for no filter.",
                        "default": "all"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Filter by years in format 'YYYY-YYYY'. Use 'all' for no filter.",
                        "default": "all"
                    },
                    "num_batsman": {
                        "type": "integer",
                        "description": "Number of batsmen to select",
                        "default": 5
                    },
                    "num_allrounder": {
                        "type": "integer",
                        "description": "Number of all-rounders to select",
                        "default": 1
                    },
                    "num_wicketkeeper": {
                        "type": "integer",
                        "description": "Number of wicket-keepers to select",
                        "default": 1
                    },
                    "num_bowler": {
                        "type": "integer",
                        "description": "Number of bowlers to select",
                        "default": 4
                    }
                },
                "required": ["match_format"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "match_scorecard",
            "description": "Get match scorecards for a specific format and optional filters",
            "parameters": {
                "type": "object",
                "properties": {
                    "match_format": {
                        "type": "string",
                        "enum": ["Test", "ODI", "T20"],
                        "description": "Cricket format"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Date range in format 'YYYY-MM-DD to YYYY-MM-DD'. Use 'all' for no filter.",
                        "default": "all"
                    },
                    "team1": {
                        "type": "string",
                        "description": "First team name (optional)",
                        "default": None
                    },
                    "team2": {
                        "type": "string",
                        "description": "Second team name (optional)",
                        "default": None
                    },
                    "venue": {
                        "type": "string",
                        "description": "Match venue (optional)",
                        "default": None
                    }
                },
                "required": ["match_format"]
            }
        }
    }
]


def clean_nan_values(data):
    """Convert NaN, inf, and -inf values to None for JSON serialization"""
    if isinstance(data, list):
        return [clean_nan_values(item) for item in data]
    elif isinstance(data, dict):
        return {k: clean_nan_values(v) for k, v in data.items()}
    elif isinstance(data, float):
        if pd.isna(data) or np.isinf(data):
            return None
        return round(data, 2) if isinstance(data, float) else data
    else:
        return data


def execute_function_call(function_name, params):
    """Execute a function based on OpenAI's tool call"""
    try:
        if function_name == "find_player":
            player_name = params.get("player_name")
            country = params.get("country", "all")
            timeframe = params.get("timeframe", "all")
            formats = params.get("formats", ["Test", "ODI", "T20"])

            # Convert timeframe string to tuple
            if timeframe != "all" and "-" in timeframe:
                years = timeframe.split("-")
                timeframe = (int(years[0]), int(years[1]))

            result = find_player(player_name, country, timeframe, formats)

            # Convert DataFrames to dict records
            agg_data = result.get("Aggregated", pd.DataFrame()).to_dict("records") if not result.get("Aggregated", pd.DataFrame()).empty else []
            match_data = result.get("Matches", pd.DataFrame()).to_dict("records") if not result.get("Matches", pd.DataFrame()).empty else []

            # Clean NaN values for JSON serialization
            agg_data = clean_nan_values(agg_data)
            match_data = clean_nan_values(match_data)

            return {
                "aggregated_count": len(agg_data),
                "matches_count": len(match_data),
                "aggregated_data": agg_data[:5],  # Limit to first 5 records
                "matches_data": match_data[:5]  # Limit to first 5 records
            }

        elif function_name == "select_best_XI":
            match_format = params.get("match_format")
            country = params.get("country", "all")
            timeframe = params.get("timeframe", "all")
            num_roles = {
                "Batsman": params.get("num_batsman", 5),
                "Allrounder": params.get("num_allrounder", 1),
                "Wicketkeeper": params.get("num_wicketkeeper", 1),
                "Bowler": params.get("num_bowler", 4)
            }

            # Convert timeframe string to tuple
            if timeframe != "all" and "-" in timeframe:
                years = timeframe.split("-")
                timeframe = (int(years[0]), int(years[1]))

            file = csv_files.get(match_format)
            result = select_best_XI(file, match_format, country, timeframe, num_roles)

            # Convert DataFrame to dict records and clean NaN values
            team_data = clean_nan_values(result.to_dict("records"))

            return {
                "team_count": len(team_data),
                "team": team_data
            }

        elif function_name == "match_scorecard":
            match_format = params.get("match_format")
            timeframe = params.get("timeframe", "all")
            team1 = params.get("team1")
            team2 = params.get("team2")
            venue = params.get("venue")

            # Convert timeframe string to tuple
            if timeframe != "all" and " to " in timeframe:
                dates = timeframe.split(" to ")
                timeframe = (dates[0].strip(), dates[1].strip())

            result = match_scorecard(match_format, timeframe, team1, team2, venue)

            # Convert DataFrame to dict records and clean NaN values
            scorecard_data = clean_nan_values(result.to_dict("records"))

            return {
                "matches_count": len(scorecard_data),
                "data": scorecard_data[:10]  # Limit to first 10 matches
            }

        else:
            return {"error": f"Unknown function: {function_name}"}

    except Exception as e:
        return {"error": str(e)}



# Flask API routes for chatbot
@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint that handles OpenAI function calling"""
    data = request.json
    user_message = data.get('message', '')
    conversation_history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # System prompt to guide the AI
    system_prompt = {
        "role": "system",
        "content": """You are a cricket statistics expert assistant. When users ask about players, teams, or cricket statistics:

1. ALWAYS use the available tools to fetch real data from the cricket database
2. For questions about "best XI" or "best team", use select_best_XI with the appropriate format and country:
   - If user mentions a country (England, India, Australia, etc.), use that country parameter
   - If user mentions a format (Test, ODI, T20), use that format
   - Always call the tool to get real rankings based on performance scores
3. For questions about specific players or comparisons:
   - Use find_player to get detailed stats for each player
   - For comparisons like "compare A to B", call find_player for BOTH players
   - Then present a side-by-side comparison of their key statistics
   - Extract player names from abbreviations (e.g., "J Root" = "Joe Root", "A Cook" = "Alastair Cook")
4. For questions about matches, use match_scorecard to get real match data
5. Present the data clearly, including relevant statistics like runs, averages, strike rates, wickets, economy rates, etc.
6. When presenting player data, include player names, teams, roles, and key performance metrics
7. Never give just opinions - always back up with actual statistics from the database
8. If you're unsure about the exact format or country name, make your best interpretation and call the tool

Always prioritize fetching real data over providing general knowledge."""
    }

    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Build messages list with system prompt
    messages_for_api = [system_prompt]
    messages_for_api.extend(conversation_history)

    try:
        # Call OpenAI with tools
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages_for_api,
            tools=TOOLS,
            tool_choice="auto"
        )

        # Check if function calling is needed
        if response.choices[0].message.tool_calls:
            assistant_message = response.choices[0].message
            tool_results = []

            # Execute each tool call
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_params = json.loads(tool_call.function.arguments)

                # Execute the function
                function_result = execute_function_call(function_name, function_params)

                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "function_name": function_name,
                    "result": function_result
                })

            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in response.choices[0].message.tool_calls]
            })

            # Add tool results to history
            for tool_result in tool_results:
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "content": json.dumps(tool_result['result'], default=str)
                })

            # Get final response from OpenAI
            final_response = client.chat.completions.create(
                model="gpt-4",
                messages=conversation_history
            )

            final_message = final_response.choices[0].message.content
            conversation_history.append({
                "role": "assistant",
                "content": final_message
            })

            return jsonify({
                "response": final_message,
                "history": conversation_history,
                "tool_calls": tool_results
            })

        else:
            # No function calling needed, just return the response
            final_message = response.choices[0].message.content
            conversation_history.append({
                "role": "assistant",
                "content": final_message
            })

            return jsonify({
                "response": final_message,
                "history": conversation_history
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Flask route
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    best_XI_html = None
    player_agg_html = None
    player_matches_html = None
    scorecard_html = None

    if request.method == 'POST':
        # --- Best XI ---
        if 'match_format' in request.form and 'num_batsman' in request.form:
            match_format = request.form.get('match_format')
            country = request.form.get('country','all')
            timeframe = request.form.get('timeframe','all')
            if timeframe != 'all' and '-' in timeframe:
                start_year, end_year = map(int, timeframe.split('-'))
                timeframe = (start_year,end_year)

            num_roles = {
                'Batsman': int(request.form.get('num_batsman',5)),
                'Allrounder': int(request.form.get('num_allrounder',1)),
                'Wicketkeeper': int(request.form.get('num_wicketkeeper',1)),
                'Bowler': int(request.form.get('num_bowler',4))
            }

            file = csv_files.get(match_format)
            best_XI = select_best_XI(file, match_format, country, timeframe, num_roles)
            best_XI_html = best_XI.to_html(classes='table table-striped', index=False)

        # --- Player search ---
        if 'player_name' in request.form and request.form.get('player_name'):
            player_name = request.form.get('player_name')
            country = request.form.get('player_country', 'all')
            timeframe = request.form.get('player_timeframe', 'all')
            formats = request.form.getlist('player_formats') or ['Test', 'ODI', 'T20']

            if timeframe != 'all' and '-' in timeframe:
                start_year, end_year = map(int, timeframe.split('-'))
                timeframe = (start_year, end_year)

            player_results = find_player(player_name, country, timeframe, formats)

            # Aggregated stats
            if not player_results['Aggregated'].empty:
                player_agg_html = player_results['Aggregated'].to_html(classes='table table-striped', index=False)

            # Match stats sorted by Weighted Impact Score descending
            if not player_results['Matches'].empty:
                if 'Weighted Impact Score' in player_results['Matches'].columns:
                    player_results['Matches'] = player_results['Matches'].sort_values(
                        by='Weighted Impact Score', ascending=False
                    ).head(10)
                    cols = [
                        'Player', 'Date', 'Format', 'Venue', 'Team', 'Opponent', 'MoM Awarded',
                        'Runs', 'Balls Faced', 'Batting Average (Match)', 'Strike Rate',
                        'Wickets', 'Runs Off Bowling', 'Balls Bowled', 'Economy',
                        'Fielding Dismissals', 'Weighted Impact Score'
                    ]
                    player_matches_df = player_results['Matches'][cols]
                    player_matches_html = player_matches_df.to_html(classes='table table-striped', index=False)

        # --- Match Scorecard ---
        if 'scorecard_match_format' in request.form:
            match_format = request.form.get('scorecard_match_format')
            timeframe = request.form.get('scorecard_timeframe', 'all')
            team1 = request.form.get('scorecard_team1')
            team2 = request.form.get('scorecard_team2')
            venue = request.form.get('scorecard_venue')

            if timeframe != 'all' and '-' in timeframe:
                start_date, end_date = timeframe.split('-')
                timeframe = (start_date.strip(), end_date.strip())

            df_scorecard = match_scorecard(match_format, timeframe, team1, team2, venue)
            if not df_scorecard.empty:
                scorecard_html = df_scorecard.to_html(classes='table table-striped', index=False)

    return render_template(
        'index.html',
        best_XI_html=best_XI_html,
        player_agg_html=player_agg_html,
        player_matches_html=player_matches_html,
        scorecard_html=scorecard_html
    )



if __name__ == '__main__':
    app.run(debug=True)
