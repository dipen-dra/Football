import streamlit as st
import sys
import os

# Add workspace to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.app_helper import init_app

def main():
    init_app("Methodology", "Detailed definitions, mathematical formulations, and modeling assumptions behind the Pressing Analytics framework.")
    
    st.markdown(r"""
    This section documents all formulas, parameters, and tactical thresholds used throughout the application to ensure transparency and reproducibility for the final-year thesis defense.
    
    ---
    
    ## 1. Passes Allowed per Defensive Action (PPDA)
    
    **PPDA** is a metric used to quantify the intensity of a team's high press. It calculates the number of passes a team allows their opponent to complete before attempting a defensive action.
    
    ### Mathematical Formulation
    
    $$\text{PPDA} = \frac{\text{Opponent Completed Passes in Zone}}{\text{Pressing Team Defensive Actions in Zone}}$$
    
    ### Parameter & Boundary Definitions:
    - **Defensive Zone:** The opponent's defensive 60% of the pitch.
      - In StatsBomb coordinates (0 to 120 yards long, 80 yards wide):
        - For the **possessing team** (opponent), passes must start at $x \le 72$.
        - For the **pressing team**, defensive actions must occur at $x \ge 48$.
    - **Opponent Completed Passes:** Completed passes only (`pass_outcome` is null/NaN).
    - **Defensive Actions:**
      - **Tackles / Challenges:** StatsBomb `Duel` events of type `Tackle` (includes both won, lost, and neutral tackles).
      - **Interceptions:** StatsBomb `Interception` events.
      - **Fouls Committed:** StatsBomb `Foul Committed` events (committed in the defined zone).
      
    *Note: A lower PPDA value signifies a more intense pressing system.*
    
    ---
    
    ## 2. Pressing Events & Spatial Thirds
    
    StatsBomb records individual pressure events (where a defender actively closes down an opponent ball carrier).
    
    ### Spatial Pitch Zones
    We segment the pitch length ($X$-axis, 0-120 yards) into three equal thirds from the pressing team's perspective:
    - **Defensive Third:** $x < 40$ yards
    - **Middle Third:** $40 \le x < 80$ yards
    - **Attacking Third:** $x \ge 80$ yards
    
    ---
    
    ## 3. Pressing Triggers Classification
    
    We categorize what event immediately preceded a pressure event within the same possession sequence:
    
    | Trigger Category | Technical Definition | Tactical Rationale |
    | :--- | :--- | :--- |
    | **Poor Touch / Miscontrol** | A `Bad Touch` or `Dispossessed` event by the opponent within the preceding 2 events. | Players close down the ball when the attacker loses clean control. |
    | **Backward Pass** | A `Pass` event by the opponent where the end $X$-coordinate is less than the start $X$-coordinate. | Backward passes indicate defensive retreat and trigger a collective forward push. |
    | **Throw-in** | A `Pass` of type `Throw-in` preceding the pressure. | Structured restarts in tight areas allow defenders to pin opponents. |
    | **Carrying / Dribbling** | A `Carry` event immediately preceding the pressure. | Defenders close down players attempting to progress the ball. |
    | **Ball Recovery / Def. Action** | A `Ball Recovery` or `Interception` by the opponent preceding the pressure. | Attacking transitions are pressed immediately before the opponent settles. |
    | **Pass Reception / Build-up** | A standard complete pass reception. | Basic press during opponent's build-up phase. |
    
    ---
    
    ## 4. Counter-Pressing Definition
    
    A pressure event is flagged as a **Counter-press** if it occurs within **5 seconds** of the pressing team losing possession of the ball.
    - Let $t_{\text{loss}}$ be the timestamp of the last event of Team A's possession.
    - A pressure event by Team A at $t_{\text{press}}$ is a counter-press if $0 \le t_{\text{press}} - t_{\text{loss}} \le 5$ seconds.
    
    ---
    
    ## 5. Machine Learning Formulations
    
    ### Model 1: Pressing Success Classifier (Event-Level)
    - **Target Variable ($Y$):** Did this pressure event lead to a ball regain by the pressing team within **5 seconds**?
    - **Features ($X$):**
      - Starting Coordinates ($x$, $y$) and distance to goal.
      - Score Differential (at the time of the event).
      - Match Minute.
      - Counter-press flag, Under Pressure flag.
      - Categorical Trigger (One-Hot Encoded).
    - **Models Evaluated:** Logistic Regression vs. XGBoost Classifier.
    
    ### Model 2: Match Outcome Predictor (Team/Match-Level)
    - **Target Variable ($Y$):** Match outcome (Win = 2, Draw = 1, Loss = 0).
    - **Features ($X$):** Team PPDA, Total Pressures, Spatial Distribution, Counter-press count, Press success rate, Possession %, xG Scored, and xG Conceded.
    - **Models Evaluated:** Multiclass Logistic Regression vs. Random Forest Classifier.
    
    ---
    
    ## 6. Analytical Limitations & Confounders
    
    > [!WARNING]
    > **Thesis Defense Critical Notes:**
    > 1. **No Off-ball Coordinates:** StatsBomb Open Data is event data, not tracking data. We only observe the location of the player pressing and the ball, not the positions of the other 20 players on the pitch. This limits our ability to evaluate defensive compactness or passing lane blockages.
    > 2. **Score Effects (Confounder):** Teams that are leading tend to drop their pressing intensity (raising their PPDA) and sit in a low block to protect the lead. Conversely, trailing teams press aggressively. This creates a feedback loop where "better" teams might show worse (higher) PPDA in certain periods of a match.
    > 3. **Causality vs. Correlation:** The models are predictive, not causal. A high pressing success rate is correlated with match outcomes, but is also heavily co-dependent on team quality, physical fitness, and game states.
    """)
    
if __name__ == "__main__":
    main()
