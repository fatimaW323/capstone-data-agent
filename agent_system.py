"""
Agent System for Autonomous Data Cleaning
Extracted from V1_allpdf_csv_chatbot.ipynb - BLOCK 2
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


# ============================================================
# AgentMemory Class
# ============================================================

class AgentMemory:
    """Shared memory/state for agent coordination."""

    def __init__(self):
        self.raw_dataframe: Optional[pd.DataFrame] = None
        self.cleaned_dataframe: Optional[pd.DataFrame] = None
        self.current_dataframe: Optional[pd.DataFrame] = None

        self.readiness_reports: List[Dict[str, Any]] = []
        self.cleaning_actions: List[Dict[str, Any]] = []
        self.insights: List[Dict[str, Any]] = []

        self.workflow_state: Dict[str, Any] = {
            "stage": "initialized",
            "ready_for_insights": False,
            "iterations": 0,
            "max_iterations": 3,
        }

        self.metrics: Dict[str, Any] = {
            "start_time": datetime.now(),
            "total_tool_calls": 0,
            "agent_calls": {},
            "errors": [],
        }

    def set_raw_data(self, df: pd.DataFrame) -> None:
        self.raw_dataframe = df.copy()
        self.current_dataframe = df.copy()
        self.workflow_state["stage"] = "profiling"

    def get_current_data(self) -> pd.DataFrame:
        if self.current_dataframe is None:
            raise ValueError("No current dataframe available")
        return self.current_dataframe.copy()

    def update_current_data(self, df: pd.DataFrame) -> None:
        self.current_dataframe = df.copy()
        self.cleaned_dataframe = df.copy()

    def add_readiness_report(self, report: Dict[str, Any]) -> None:
        self.readiness_reports.append(report)

    def add_cleaning_action(self, action: Dict[str, Any]) -> None:
        self.cleaning_actions.append(action)

    def add_insight(self, insight: Dict[str, Any]) -> None:
        self.insights.append(insight)


# ============================================================
# DataReadinessAgent
# ============================================================

class DataReadinessAgent:
    """Assesses data quality and readiness."""

    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.name = "DataReadinessAgent"

    def profile_dataset(self) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        
        df = self.memory.get_current_data()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        # Missing values
        report["missing_values"] = self._analyze_missing_values(df)
        
        # Data types
        report["data_types"] = self._analyze_data_types(df)
        
        # Duplicates
        report["duplicates"] = self._check_duplicates(df)
        
        # Outliers
        report["outliers"] = self._detect_outliers(df)
        
        # Calculate readiness score
        report["readiness_score"] = self._calculate_readiness_score(report)
        
        # Add to memory
        self.memory.add_readiness_report(report)
        
        return report

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        result = {}
        for col in df.columns:
            if missing[col] > 0:
                result[col] = {
                    "count": int(missing[col]),
                    "percentage": float(missing_pct[col])
                }
        
        return result

    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types."""
        type_issues = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if should be numeric
                numeric_pct = pd.to_numeric(df[col], errors='coerce').notna().sum() / len(df)
                if numeric_pct > 0.8:
                    type_issues.append({
                        "column": col,
                        "current_type": "object",
                        "suggested_type": "numeric",
                        "confidence": numeric_pct
                    })
        
        return {"issues": type_issues}

    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate rows."""
        dup_count = df.duplicated().sum()
        
        return {
            "count": int(dup_count),
            "percentage": float(dup_count / len(df) * 100) if len(df) > 0 else 0
        }

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        outliers = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    "count": int(outlier_count),
                    "percentage": float(outlier_count / len(df) * 100)
                }
        
        return outliers

    def _calculate_readiness_score(self, report: Dict[str, Any]) -> float:
        """Calculate overall data readiness score (0-1)."""
        
        score = 1.0
        
        # Penalty for missing values (max -0.3)
        if report["missing_values"]:
            avg_missing_pct = sum(v["percentage"] for v in report["missing_values"].values()) / len(report["missing_values"])
            penalty = min(avg_missing_pct / 100 * 0.3, 0.3)
            score -= penalty
        
        # Penalty for duplicates (max -0.2)
        dup_pct = report["duplicates"]["percentage"]
        penalty = min(dup_pct / 100 * 0.2, 0.2)
        score -= penalty
        
        # Penalty for type issues (max -0.15)
        type_issues = len(report["data_types"]["issues"])
        if type_issues > 0:
            penalty = min(type_issues * 0.05, 0.15)
            score -= penalty
        
        # Penalty for outliers (max -0.1)
        if report["outliers"]:
            avg_outlier_pct = sum(v["percentage"] for v in report["outliers"].values()) / len(report["outliers"])
            penalty = min(avg_outlier_pct / 100 * 0.1, 0.1)
            score -= penalty
        
        return max(0.0, min(1.0, score))


# ============================================================
# DataCleaningAgent
# ============================================================

class DataCleaningAgent:
    """Cleans data based on readiness report."""

    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.name = "DataCleaningAgent"

    def clean_dataset(self) -> Dict[str, Any]:
        """Apply cleaning actions."""
        
        df = self.memory.get_current_data()
        actions_taken = []
        
        # Get latest readiness report
        if not self.memory.readiness_reports:
            return {"actions": [], "status": "No readiness report available"}
        
        report = self.memory.readiness_reports[-1]
        
        # Handle missing values
        df, missing_actions = self._handle_missing_values(df, report)
        actions_taken.extend(missing_actions)
        
        # Remove duplicates
        df, dup_actions = self._remove_duplicates(df, report)
        actions_taken.extend(dup_actions)
        
        # Fix data types
        df, type_actions = self._fix_data_types(df, report)
        actions_taken.extend(type_actions)
        
        # Update memory
        self.memory.update_current_data(df)
        
        action_record = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "actions": actions_taken
        }
        self.memory.add_cleaning_action(action_record)
        
        return action_record

    def _handle_missing_values(self, df: pd.DataFrame, report: Dict) -> tuple:
        """Handle missing values."""
        actions = []
        
        for col, info in report.get("missing_values", {}).items():
            if col not in df.columns:
                continue
            
            # Drop column if >70% missing
            if info["percentage"] > 70:
                df = df.drop(columns=[col])
                actions.append(f"Dropped column '{col}' ({info['percentage']:.1f}% missing)")
            else:
                # Impute based on type
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                    actions.append(f"Filled {info['count']} missing in '{col}' with median")
                else:
                    if not df[col].mode().empty:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                        actions.append(f"Filled {info['count']} missing in '{col}' with mode")
        
        return df, actions

    def _remove_duplicates(self, df: pd.DataFrame, report: Dict) -> tuple:
        """Remove duplicate rows."""
        actions = []
        
        dup_count = report["duplicates"]["count"]
        if dup_count > 0:
            df = df.drop_duplicates()
            actions.append(f"Removed {dup_count} duplicate rows")
        
        return df, actions

    def _fix_data_types(self, df: pd.DataFrame, report: Dict) -> tuple:
        """Fix data type issues."""
        actions = []
        
        for issue in report["data_types"]["issues"]:
            col = issue["column"]
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                actions.append(f"Converted '{col}' to numeric")
        
        return df, actions


# ============================================================
# InsightDiscoveryAgent
# ============================================================

class InsightDiscoveryAgent:
    """Discovers insights from cleaned data."""

    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.name = "InsightDiscoveryAgent"

    def discover_insights(self) -> Dict[str, Any]:
        """Discover patterns and insights."""
        
        df = self.memory.get_current_data()
        
        insights = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "key_findings": []
        }
        
        # Basic statistics
        insights["statistics"] = self._generate_statistics(df)
        
        # Correlations
        insights["correlations"] = self._find_correlations(df)
        
        # Distribution analysis
        insights["distributions"] = self._analyze_distributions(df)
        
        # Generate key findings
        insights["key_findings"] = self._generate_key_findings(insights, df)
        
        # Add to memory
        self.memory.add_insight(insights)
        
        return insights

    def _generate_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate basic statistics."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        stats = {}
        for col in numeric_cols[:5]:  # Limit to first 5
            stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
        
        return stats

    def _find_correlations(self, df: pd.DataFrame) -> List[Dict]:
        """Find strong correlations."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            return []
        
        corr_matrix = df[numeric_cols].corr()
        
        correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    correlations.append({
                        "feature1": numeric_cols[i],
                        "feature2": numeric_cols[j],
                        "correlation": float(corr_val)
                    })
        
        return sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)[:5]

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict:
        """Analyze distributions."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        distributions = {}
        for col in numeric_cols[:5]:
            skewness = float(df[col].skew())
            distributions[col] = {
                "skewness": skewness,
                "distribution_type": "normal" if abs(skewness) < 0.5 else "skewed"
            }
        
        return distributions

    def _generate_key_findings(self, insights: Dict, df: pd.DataFrame) -> List[str]:
        """Generate human-readable key findings."""
        findings = []
        
        findings.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
        
        if insights["correlations"]:
            top_corr = insights["correlations"][0]
            findings.append(
                f"Strong correlation ({top_corr['correlation']:.2f}) between "
                f"{top_corr['feature1']} and {top_corr['feature2']}"
            )
        
        return findings


# ============================================================
# OrchestratorAgent
# ============================================================

class OrchestratorAgent:
    """Coordinates all agents in the pipeline."""

    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.readiness_agent = DataReadinessAgent(memory)
        self.cleaning_agent = DataCleaningAgent(memory)
        self.insight_agent = InsightDiscoveryAgent(memory)
        self.name = "OrchestratorAgent"

    def run_pipeline(self, target_score: float = 0.8, max_iterations: int = 3, verbose: bool = True) -> Dict[str, Any]:
        """Run the complete autonomous pipeline."""
        
        if verbose:
            print(f"🚀 Starting Autonomous Data Cleaning Pipeline")
            print("=" * 50)
        
        # Stage 1: Initial Assessment
        if verbose:
            print("\n📊 Stage 1: Assessing data quality...")
        
        initial_report = self.readiness_agent.profile_dataset()
        initial_score = initial_report["readiness_score"]
        
        if verbose:
            print(f"   Initial quality score: {initial_score:.2f}/1.00")
        
        # Stage 2: Iterative Cleaning
        current_score = initial_score
        iteration = 0
        
        while current_score < target_score and iteration < max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n🔧 Stage 2.{iteration}: Cleaning data (iteration {iteration}/{max_iterations})...")
            
            # Clean
            cleaning_result = self.cleaning_agent.clean_dataset()
            
            if verbose:
                for action in cleaning_result["actions"]:
                    print(f"   ✓ {action}")
            
            # Re-assess
            new_report = self.readiness_agent.profile_dataset()
            new_score = new_report["readiness_score"]
            
            if verbose:
                print(f"   Quality score: {new_score:.2f}/1.00 (Δ +{new_score - current_score:.2f})")
            
            # Check if improved
            if new_score <= current_score:
                if verbose:
                    print("   ⚠️ No improvement detected, stopping iterations")
                break
            
            current_score = new_score
        
        # Stage 3: Insight Discovery
        if verbose:
            print(f"\n💡 Stage 3: Discovering insights...")
        
        insights = self.insight_agent.discover_insights()
        
        if verbose:
            print(f"   Found {len(insights['correlations'])} strong correlations")
            print(f"   Generated {len(insights['key_findings'])} key findings")
        
        # Final results
        final_report = {
            "initial_readiness_score": initial_score,
            "readiness_score": current_score,
            "iterations": iteration,
            "target_reached": current_score >= target_score,
            "cleaning_actions": self.memory.cleaning_actions,
            "insights": [insights],
            "readiness_reports": self.memory.readiness_reports,
            "cleaned_data": self.memory.get_current_data()
        }
        
        if verbose:
            print("\n" + "=" * 50)
            print(f"✅ Pipeline Complete!")
            print(f"   Final Score: {current_score:.2f}/1.00")
            print(f"   Improvement: +{current_score - initial_score:.2f}")
            print(f"   Target Met: {'Yes ✓' if final_report['target_reached'] else 'No ✗'}")
        
        return final_report


# ============================================================
# Main Pipeline Function
# ============================================================

def run_autonomous_pipeline(
    df: pd.DataFrame,
    target_score: float = 0.8,
    max_iterations: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the complete autonomous data cleaning pipeline.
    
    Args:
        df: Input dataframe
        target_score: Target quality score (0-1)
        max_iterations: Maximum cleaning iterations
        verbose: Print progress
    
    Returns:
        Dictionary with results
    """
    
    # Initialize memory and orchestrator
    memory = AgentMemory()
    memory.set_raw_data(df)
    
    orchestrator = OrchestratorAgent(memory)
    
    # Run pipeline
    results = orchestrator.run_pipeline(
        target_score=target_score,
        max_iterations=max_iterations,
        verbose=verbose
    )
    
    return results
