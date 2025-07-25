# Elite Baseball Training - Streamlit Web App
# Save this as: streamlit_app.py
# Run with: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
import glob

# Page configuration
st.set_page_config(
    page_title="Elite Baseball Training - Program Analysis",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .flag-high { background-color: #ffebee; }
    .flag-medium { background-color: #fff3e0; }
    .flag-low { background-color: #e8f5e8; }
</style>
""", unsafe_allow_html=True)

class BaseballAnalyzer:
    def __init__(self):
        self.df = None
        self.flag_criteria = {
            'AVG_low': 0.250,
            'SLG_low': 0.350,
            'OPS_low': 0.700,
            'GB_high': 66.0,
            'K_high': 30.0
        }
    
    def load_data_from_folder(self, data_folder="data"):
        """Load and process CSV files from local data folder"""
        if not os.path.exists(data_folder):
            st.error(f"❌ Data folder '{data_folder}' not found!")
            st.info(f"Please create a '{data_folder}' folder and add your CSV files there.")
            return False
        
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
        
        if not csv_files:
            st.error(f"❌ No CSV files found in '{data_folder}' folder!")
            st.info(f"Please add your GameChanger CSV files to the '{data_folder}' folder.")
            return False
        
        all_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, csv_file in enumerate(csv_files):
            try:
                # Update progress
                progress = (i + 1) / len(csv_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {os.path.basename(csv_file)}...")
                
                # Read CSV with special handling for GameChanger format
                # Read all lines first
                with open(csv_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Find the header line (contains "Number", "Last", "First", etc.)
                header_line_idx = None
                for idx, line in enumerate(lines):
                    # Check for both quoted and unquoted header patterns
                    has_number = ('Number' in line or '"Number"' in line)
                    has_last = ('Last' in line or '"Last"' in line)
                    has_first = ('First' in line or '"First"' in line)
                    
                    if has_number and has_last and has_first:
                        header_line_idx = idx
                        break
                
                if header_line_idx is None:
                    st.warning(f"⚠️ Could not find proper headers in {os.path.basename(csv_file)}")
                    continue
                
                # Read from header line onwards
                df = pd.read_csv(csv_file, skiprows=header_line_idx, header=0)
                
                # Extract team info from filename
                team_info = os.path.basename(csv_file).replace('.csv', '').replace('Stats', '').strip()
                
                # Clean team names
                team_info = self.clean_team_name(team_info)
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Focus on batting data (first 54 columns or until pitching section)
                batting_cols = []
                for col in df.columns:
                    if 'IP' in col and len(batting_cols) > 20:  # IP usually starts pitching section
                        break
                    batting_cols.append(col)
                
                # Keep only batting columns (up to about 54 columns or before pitching starts)
                if len(batting_cols) > 54:
                    batting_cols = batting_cols[:54]
                
                df = df[batting_cols]
                
                # Add team information
                df['Team'] = team_info
                df['Source_File'] = os.path.basename(csv_file)
                
                # Clean player data
                if 'Last' in df.columns and 'First' in df.columns:
                    df = df.dropna(subset=['Last', 'First'])
                    df = df[df['Last'].notna() & (df['Last'] != '')]
                    df = df[df['Last'] != 'Last']  # Remove any duplicate header rows
                
                # Convert numeric columns with robust error handling
                numeric_cols = ['GP', 'PA', 'AB', 'AVG', 'OBP', 'OPS', 'SLG', 'H', '1B', '2B', 
                              '3B', 'HR', 'RBI', 'R', 'BB', 'SO', 'HBP', 'GB%', 'LD%', 'FB%', 'BABIP', 'TB', 'SF']
                
                for col in numeric_cols:
                    if col in df.columns:
                        # Convert to string first, then handle special values
                        df[col] = df[col].astype(str)
                        # Replace problematic values
                        df[col] = df[col].replace(['-', '', 'nan', 'NaN', '#DIV/0!', '#N/A', 'inf', '-inf'], np.nan)
                        # Convert to numeric, coercing errors to NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Calculate K% with better error handling
                if 'SO' in df.columns and 'PA' in df.columns:
                    # Ensure both columns are numeric
                    df['SO'] = pd.to_numeric(df['SO'], errors='coerce')
                    df['PA'] = pd.to_numeric(df['PA'], errors='coerce')
                    
                    df['K%'] = np.where(
                        (df['PA'] > 0) & df['PA'].notna() & df['SO'].notna(),
                        (df['SO'] / df['PA']) * 100,
                        np.nan
                    )
                
                # Filter players with at least 20 AB
                if 'AB' in df.columns:
                    df['AB'] = pd.to_numeric(df['AB'], errors='coerce')
                    initial_count = len(df)
                    df = df[df['AB'] >= 20]
                    filtered_count = initial_count - len(df)
                
                all_data.append(df)
                
            except Exception as e:
                st.warning(f"⚠️ Error processing {os.path.basename(csv_file)}: {str(e)}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if all_data:
            self.df = pd.concat(all_data, ignore_index=True)
            self.df['Age_Group'] = self.df['Team'].apply(self.extract_age_group)
            self.flag_players()
            
            st.success(f"✅ Successfully loaded {len(csv_files)} CSV files with {len(self.df)} players!")
            return True
        
        st.error("❌ No valid data could be loaded from CSV files.")
        return False
    
    def clean_team_name(self, team_name):
        """Clean team names to show only coach names"""
        name = team_name
        
        # Remove prefixes
        prefixes = ["Elite Baseball Training ", "Elite Baseball - ", "Elite Baseball-", "Elite Baseball ", "Elite ", "EBT "]
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        
        # Remove suffixes
        suffixes = [" Summer 2025 Stats", " Spring 2025 Stats", " Summer 2025", " Spring 2025", " Stats"]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        # Extract coach name and age group
        age_match = re.search(r'(\d+U)', name)
        age_group = age_match.group(1) if age_match else ""
        coach_name = re.sub(r'\s*\d+U\s*', ' ', name).strip()
        coach_name = re.sub(r'\s+', ' ', coach_name)
        
        if coach_name and age_group:
            return f"{coach_name} {age_group}"
        elif coach_name:
            return coach_name
        elif age_group:
            return age_group
        else:
            return name
    
    def extract_age_group(self, team_name):
        """Extract age group from team name"""
        # First try to find standard age pattern (e.g., "16U", "15U")
        age_match = re.search(r'(\d+)U', team_name)
        if age_match:
            return f"{age_match.group(1)}U"
        
        # Handle special cases
        if 'Ayeski' in team_name:
            return "17U"  # Ayeski is a 17U team
        
        # Try to find just age numbers in context
        age_match = re.search(r'\b(\d{2})\b', team_name)
        if age_match:
            age = int(age_match.group(1))
            if 10 <= age <= 18:
                return f"{age}U"
        
        return "Unknown"
    
    def flag_players(self):
        """Flag players based on criteria"""
        if self.df is None:
            return
        
        # Create flags
        self.df['Flag_AVG_Low'] = (self.df['AVG'] < self.flag_criteria['AVG_low']) & self.df['AVG'].notna()
        self.df['Flag_SLG_Low'] = (self.df['SLG'] < self.flag_criteria['SLG_low']) & self.df['SLG'].notna()
        self.df['Flag_OPS_Low'] = (self.df['OPS'] < self.flag_criteria['OPS_low']) & self.df['OPS'].notna()
        self.df['Flag_GB_High'] = (self.df['GB%'] > self.flag_criteria['GB_high']) & self.df['GB%'].notna()
        self.df['Flag_K_High'] = (self.df['K%'] > self.flag_criteria['K_high']) & self.df['K%'].notna()
        
        # Count total flags
        flag_cols = ['Flag_AVG_Low', 'Flag_SLG_Low', 'Flag_OPS_Low', 'Flag_GB_High', 'Flag_K_High']
        self.df['Total_Flags'] = self.df[flag_cols].sum(axis=1)

def main():
    st.markdown('<h1 class="main-header">⚾ Elite Baseball Training - Program Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = BaseballAnalyzer()
        st.session_state.data_loaded = False
    
    # Sidebar for data management
    with st.sidebar:
        st.header("📁 Data Management")
        
        # Data folder info
        data_folder = st.text_input("Data Folder Path", value="data", help="Folder containing your CSV files")
        
        if st.button("🔄 Load/Reload Data"):
            with st.spinner("Loading data from folder..."):
                if st.session_state.analyzer.load_data_from_folder(data_folder):
                    st.session_state.data_loaded = True
                    st.rerun()
                else:
                    st.session_state.data_loaded = False
        
        # Show data folder contents
        if os.path.exists(data_folder):
            csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
            st.subheader("📋 Files in Data Folder")
            if csv_files:
                for csv_file in csv_files:
                    st.text(f"📄 {os.path.basename(csv_file)}")
            else:
                st.info("No CSV files found in data folder")
        else:
            st.info(f"Create a '{data_folder}' folder and add your CSV files")
        
        # Flag criteria adjustment
        if st.session_state.data_loaded and st.session_state.analyzer.df is not None:
            st.header("⚙️ Flag Criteria")
            st.session_state.analyzer.flag_criteria['AVG_low'] = st.number_input("AVG < ", value=0.250, step=0.001, format="%.3f")
            st.session_state.analyzer.flag_criteria['SLG_low'] = st.number_input("SLG < ", value=0.350, step=0.001, format="%.3f")
            st.session_state.analyzer.flag_criteria['OPS_low'] = st.number_input("OPS < ", value=0.700, step=0.001, format="%.3f")
            st.session_state.analyzer.flag_criteria['GB_high'] = st.number_input("GB% > ", value=66.0, step=0.1, format="%.1f")
            st.session_state.analyzer.flag_criteria['K_high'] = st.number_input("K% > ", value=30.0, step=0.1, format="%.1f")
            
            if st.button("🔄 Update Flags"):
                st.session_state.analyzer.flag_players()
                st.rerun()
    
    # Main content
    if not st.session_state.data_loaded or st.session_state.analyzer.df is None:
        st.info("👆 Please load data using the sidebar to get started.")
        return
    
    df = st.session_state.analyzer.df
    
    # Program Overview Metrics
    st.header("📊 Program Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Players", len(df))
    with col2:
        st.metric("Total Teams", df['Team'].nunique())
    with col3:
        flagged_count = (df['Total_Flags'] > 0).sum()
        st.metric("Players with Flags", flagged_count, f"{flagged_count/len(df)*100:.1f}%")
    with col4:
        st.metric("Program AVG", f"{df['AVG'].mean():.3f}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Dashboard", "🚩 Flagged Players", "👥 Team Analysis", "📋 Individual Lookup", "📊 Statistics"])
    
    with tab1:
        create_dashboard_tab(df)
    
    with tab2:
        create_flagged_players_tab(df)
    
    with tab3:
        create_team_analysis_tab(df)
    
    with tab4:
        create_individual_lookup_tab(df)
    
    with tab5:
        create_statistics_tab(df)

def create_dashboard_tab(df):
    """Create the main dashboard tab"""
    st.header("Program Analysis Dashboard")
    
    # Flag distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Player Distribution by Flag Count")
        flag_dist = df['Total_Flags'].value_counts().sort_index()
        fig = px.bar(x=flag_dist.index, y=flag_dist.values, 
                    labels={'x': 'Number of Flags', 'y': 'Number of Players'},
                    color=flag_dist.index, 
                    color_continuous_scale='RdYlGn_r')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Most Common Flag Types")
        flag_types = {
            'AVG < .250': df['Flag_AVG_Low'].sum(),
            'SLG < .350': df['Flag_SLG_Low'].sum(),
            'OPS < .700': df['Flag_OPS_Low'].sum(),
            'GB% > 66%': df['Flag_GB_High'].sum(),
            'K% > 30%': df['Flag_K_High'].sum()
        }
        flag_df = pd.DataFrame(list(flag_types.items()), columns=['Flag Type', 'Count'])
        fig = px.bar(flag_df, x='Count', y='Flag Type', orientation='h',
                    color='Count', color_continuous_scale='Reds')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance distributions
    st.subheader("Performance Distributions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(df, x='AVG', nbins=30, title='Batting Average Distribution')
        fig.add_vline(x=0.250, line_dash="dash", line_color="red", annotation_text="Flag Line")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='OPS', nbins=30, title='OPS Distribution')
        fig.add_vline(x=0.700, line_dash="dash", line_color="red", annotation_text="Flag Line")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(df, x='K%', nbins=30, title='Strikeout Rate Distribution')
        fig.add_vline(x=30.0, line_dash="dash", line_color="red", annotation_text="Flag Line")
        st.plotly_chart(fig, use_container_width=True)

def create_flagged_players_tab(df):
    """Create the flagged players tab"""
    st.header("🚩 Flagged Players Analysis")
    
    flagged_df = df[df['Total_Flags'] > 0]
    
    if len(flagged_df) == 0:
        st.success("🎉 No players are currently flagged! Great job!")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_flags = st.selectbox("Minimum Flags", options=[1, 2, 3, 4, 5], index=0)
    with col2:
        teams = ['All Teams'] + sorted(df['Team'].unique().tolist())
        selected_team = st.selectbox("Team", options=teams)
    with col3:
        age_groups = ['All Ages'] + sorted(df['Age_Group'].unique().tolist())
        selected_age = st.selectbox("Age Group", options=age_groups)
    
    # Apply filters
    filtered_df = flagged_df[flagged_df['Total_Flags'] >= min_flags]
    if selected_team != 'All Teams':
        filtered_df = filtered_df[filtered_df['Team'] == selected_team]
    if selected_age != 'All Ages':
        filtered_df = filtered_df[filtered_df['Age_Group'] == selected_age]
    
    # Display results
    st.subheader(f"Flagged Players ({len(filtered_df)} players)")
    
    if len(filtered_df) > 0:
        # Prepare display dataframe
        display_df = filtered_df[['First', 'Last', 'Team', 'AB', 'AVG', 'OPS', 'K%', 'GB%', 'Total_Flags']].copy()
        display_df = display_df.sort_values(['Total_Flags', 'OPS'], ascending=[False, True])
        
        # Add flag details
        flag_details = []
        for idx, row in display_df.iterrows():
            flags = []
            if filtered_df.loc[idx, 'Flag_AVG_Low']:
                flags.append("AVG")
            if filtered_df.loc[idx, 'Flag_SLG_Low']:
                flags.append("SLG")
            if filtered_df.loc[idx, 'Flag_OPS_Low']:
                flags.append("OPS")
            if filtered_df.loc[idx, 'Flag_GB_High']:
                flags.append("GB%")
            if filtered_df.loc[idx, 'Flag_K_High']:
                flags.append("K%")
            flag_details.append(", ".join(flags))
        
        display_df['Issues'] = flag_details
        
        # Display dataframe
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Flagged Players CSV",
            data=csv,
            file_name="flagged_players.csv",
            mime="text/csv"
        )
    else:
        st.info("No players match the selected criteria.")

def create_team_analysis_tab(df):
    """Create the team analysis tab"""
    st.header("👥 Team Analysis")
    
    # Team summary with weighted calculations
    def calculate_weighted_team_stats(team_df):
        """Calculate team stats weighted by at-bats"""
        # Ensure all values are numeric and handle NaN values
        def safe_sum(column_name):
            if column_name in team_df.columns:
                series = pd.to_numeric(team_df[column_name], errors='coerce')
                return series.fillna(0).sum()
            return 0
        
        total_ab = safe_sum('AB')
        total_pa = safe_sum('PA')
        total_hits = safe_sum('H')
        total_bb = safe_sum('BB')
        total_so = safe_sum('SO')
        total_hbp = safe_sum('HBP')
        total_sf = safe_sum('SF')
        
        # Calculate total bases
        if 'TB' in team_df.columns:
            total_tb = safe_sum('TB')
        else:
            # Calculate total bases from individual SLG and AB
            slg_series = pd.to_numeric(team_df['SLG'], errors='coerce').fillna(0)
            ab_series = pd.to_numeric(team_df['AB'], errors='coerce').fillna(0)
            total_tb = (slg_series * ab_series).sum()
        
        # Weighted calculations with safe division
        team_avg = total_hits / total_ab if total_ab > 0 else 0
        
        # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
        obp_denominator = total_ab + total_bb + total_hbp + total_sf
        team_obp = (total_hits + total_bb + total_hbp) / obp_denominator if obp_denominator > 0 else 0
        
        team_slg = total_tb / total_ab if total_ab > 0 else 0
        team_ops = team_obp + team_slg
        team_k_pct = (total_so / total_pa * 100) if total_pa > 0 else 0
        
        return {
            'Team_AVG': team_avg,
            'Team_OBP': team_obp, 
            'Team_SLG': team_slg,
            'Team_OPS': team_ops,
            'Team_K_Pct': team_k_pct,
            'Total_AB': int(total_ab),
            'Total_PA': int(total_pa),
            'Total_H': int(total_hits),
            'Total_BB': int(total_bb),
            'Total_SO': int(total_so),
            'Total_TB': int(total_tb)
        }
    
    # Calculate weighted team stats
    team_stats_list = []
    for team in df['Team'].unique():
        team_df = df[df['Team'] == team]
        stats = calculate_weighted_team_stats(team_df)
        stats['Team'] = team
        stats['Total_Players'] = len(team_df)
        stats['Total_Flags'] = team_df['Total_Flags'].sum()
        stats['Players_Flagged'] = (team_df['Total_Flags'] > 0).sum()
        stats['Flag_Rate'] = (stats['Players_Flagged'] / stats['Total_Players'] * 100) if stats['Total_Players'] > 0 else 0
        team_stats_list.append(stats)
    
    team_summary = pd.DataFrame(team_stats_list).set_index('Team')
    team_summary = team_summary.round(3)
    
    # Team selector
    selected_team = st.selectbox("Select Team for Detailed Analysis", 
                                options=['Overview'] + sorted(df['Team'].unique().tolist()))
    
    if selected_team == 'Overview':
        st.subheader("Team Overview")
        
        # Sort by OPS
        team_display = team_summary.sort_values('Team_OPS', ascending=False)
        
        # Display key columns
        display_cols = ['Total_Players', 'Total_AB', 'Team_AVG', 'Team_OPS', 'Team_K_Pct', 'Players_Flagged', 'Flag_Rate']
        st.dataframe(team_display[display_cols], use_container_width=True)
        
        # Team OPS chart
        fig = px.bar(team_display.reset_index(), x='Team_OPS', y='Team', 
                    orientation='h', title='Team OPS Rankings',
                    color='Team_OPS', color_continuous_scale='RdYlGn')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Individual team analysis
        team_data = df[df['Team'] == selected_team]
        team_stats = calculate_weighted_team_stats(team_data)
        
        st.subheader(f"Team Analysis: {selected_team}")
        
        # Show team stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Players", len(team_data))
            st.metric("Total At-Bats", int(team_stats['Total_AB']))
        with col2:
            flagged = (team_data['Total_Flags'] > 0).sum()
            st.metric("Flagged Players", flagged, f"{flagged/len(team_data)*100:.1f}%")
        with col3:
            st.metric("Team AVG", f"{team_stats['Team_AVG']:.3f}")
            st.metric("Team OBP", f"{team_stats['Team_OBP']:.3f}")
        with col4:
            st.metric("Team OPS", f"{team_stats['Team_OPS']:.3f}")
            st.metric("Team K%", f"{team_stats['Team_K_Pct']:.1f}%")
        
        # Player list with contribution info
        st.subheader("Team Roster")
        roster_df = team_data[['First', 'Last', 'AB', 'PA', 'AVG', 'OPS', 'K%', 'Total_Flags']].copy()
        roster_df = roster_df.sort_values('AB', ascending=False)
        
        # Add contribution percentages
        roster_df['AB_Share'] = (roster_df['AB'] / team_stats['Total_AB'] * 100).round(1)
        roster_df['PA_Share'] = (roster_df['PA'] / team_stats['Total_PA'] * 100).round(1)
        
        st.dataframe(roster_df, use_container_width=True)

def create_individual_lookup_tab(df):
    """Create the individual player lookup tab"""
    st.header("📋 Individual Player Lookup")
    
    # Player search
    players = df['First'] + ' ' + df['Last'] + ' (' + df['Team'] + ')'
    selected_player = st.selectbox("Select Player", options=sorted(players.tolist()))
    
    if selected_player:
        # Parse selection
        player_name = selected_player.split(' (')[0]
        first_name, last_name = player_name.split(' ', 1)
        
        # Find player
        player_data = df[(df['First'] == first_name) & (df['Last'] == last_name)]
        
        if len(player_data) > 0:
            player = player_data.iloc[0]
            
            st.subheader(f"Player Profile: {player['First']} {player['Last']}")
            
            # Player info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Team Information**")
                st.write(f"Team: {player['Team']}")
                st.write(f"Age Group: {player['Age_Group']}")
                st.write(f"Games Played: {player['GP']}")
                st.write(f"At Bats: {player['AB']}")
            
            with col2:
                st.markdown("**Performance Metrics**")
                st.write(f"Batting Average: {player['AVG']:.3f}")
                st.write(f"On-Base Percentage: {player['OBP']:.3f}")
                st.write(f"Slugging Percentage: {player['SLG']:.3f}")
                st.write(f"OPS: {player['OPS']:.3f}")
                st.write(f"Strikeout Rate: {player['K%']:.1f}%")
            
            # Flag status
            st.markdown("**Flag Status**")
            if player['Total_Flags'] == 0:
                st.success("✅ No flags - performing well!")
            else:
                st.warning(f"⚠️ {int(player['Total_Flags'])} flag(s) identified")
                
                flags = []
                if player['Flag_AVG_Low']:
                    flags.append("🔴 Low Batting Average")
                if player['Flag_SLG_Low']:
                    flags.append("🟠 Low Slugging Percentage")
                if player['Flag_OPS_Low']:
                    flags.append("🟡 Low OPS")
                if player['Flag_GB_High']:
                    flags.append("🟤 High Ground Ball Rate")
                if player['Flag_K_High']:
                    flags.append("🟣 High Strikeout Rate")
                
                for flag in flags:
                    st.write(flag)

def create_statistics_tab(df):
    """Create the statistics tab"""
    st.header("📊 Program Statistics")
    
    # Overall statistics
    st.subheader("Program Performance Summary")
    
    stats_data = {
        'Metric': ['Batting Average', 'On-Base Percentage', 'Slugging Percentage', 'OPS', 'Strikeout Rate'],
        'Program Average': [
            f"{df['AVG'].mean():.3f}",
            f"{df['OBP'].mean():.3f}",
            f"{df['SLG'].mean():.3f}",
            f"{df['OPS'].mean():.3f}",
            f"{df['K%'].mean():.1f}%"
        ],
        'Best': [
            f"{df['AVG'].max():.3f}",
            f"{df['OBP'].max():.3f}",
            f"{df['SLG'].max():.3f}",
            f"{df['OPS'].max():.3f}",
            f"{df['K%'].min():.1f}%"
        ],
        'Worst': [
            f"{df['AVG'].min():.3f}",
            f"{df['OBP'].min():.3f}",
            f"{df['SLG'].min():.3f}",
            f"{df['OPS'].min():.3f}",
            f"{df['K%'].max():.1f}%"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.table(stats_df)
    
    # Age group analysis
    st.subheader("Performance by Age Group")
    
    age_stats = df.groupby('Age_Group').agg({
        'AVG': 'mean',
        'OPS': 'mean',
        'K%': 'mean',
        'Total_Flags': ['count', 'sum', 'mean']
    }).round(3)
    
    age_stats.columns = ['Avg_AVG', 'Avg_OPS', 'Avg_K_Pct', 'Total_Players', 'Total_Flags', 'Avg_Flags']
    
    st.dataframe(age_stats, use_container_width=True)
    
    # Download full dataset
    st.subheader("Export Data")
    
    if st.button("📥 Download Complete Dataset"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="complete_baseball_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
