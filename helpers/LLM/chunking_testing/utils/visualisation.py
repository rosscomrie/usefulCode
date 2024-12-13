import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import numpy as np

class ChunkingVisualizer:
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        self.setup_style()
    
    def setup_style(self):
        """Set up the plotting style."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
    
    def plot_metric_comparison(self, metric: str, title: str = None):
        """Plot comparison of a specific metric across different chunking methods."""
        plt.figure(figsize=(12, 6))
        
        # Create grouped bar plot
        ax = sns.barplot(
            data=self.results_df,
            x='Configuration',
            y=metric,
            hue='Type',
            dodge=True
        )
        
        # Customize plot
        plt.xticks(rotation=45, ha='right')
        plt.title(title or f'Comparison of {metric} Across Chunking Methods')
        plt.tight_layout()
        
        return ax
    
    def plot_chunk_size_distribution(self):
        """Plot chunk size distribution for each method."""
        plt.figure(figsize=(12, 6))
        
        ax = sns.scatterplot(
            data=self.results_df,
            x='Number of Chunks',
            y='Avg Chunk Length',
            hue='Type',
            style='Type',
            s=100
        )
        
        # Add labels for each point
        for idx, row in self.results_df.iterrows():
            plt.annotate(
                row['Configuration'],
                (row['Number of Chunks'], row['Avg Chunk Length']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        plt.title('Chunk Size Distribution Across Methods')
        plt.tight_layout()
        
        return ax
    
    def plot_performance_radar(self, methods: List[str] = None):
        """Create a radar plot comparing key metrics for selected methods."""
        metrics = ['Avg Similarity', 'Max Similarity', 'Top-K Accuracy', 
                  'MRR', 'Precision@K']
        
        if methods is None:
            methods = self.results_df['Configuration'].unique()
        
        # Filter data for selected methods
        data = self.results_df[self.results_df['Configuration'].isin(methods)]
        
        # Number of metrics
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for idx, method in enumerate(methods):
            method_data = data[data['Configuration'] == method]
            values = [method_data[metric].iloc[0] for metric in metrics]
            values = np.concatenate((values, [values[0]]))  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        plt.title('Performance Metrics Comparison')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        
        return ax
    
    def plot_overview_dashboard(self, save_path: str = None):
        """Create a comprehensive dashboard of all visualizations."""
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid for subplots
        gs = fig.add_gridspec(2, 2)
        
        # Plot chunk size distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self.results_df.plot(
            kind='scatter',
            x='Number of Chunks',
            y='Avg Chunk Length',
            c='Type',
            cmap='Set2',
            ax=ax1
        )
        ax1.set_title('Chunk Size Distribution')
        
        # Plot average similarity comparison
        ax2 = fig.add_subplot(gs[0, 1])
        sns.barplot(
            data=self.results_df,
            x='Configuration',
            y='Avg Similarity',
            hue='Type',
            ax=ax2
        )
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_title('Average Similarity Comparison')
        
        # Plot precision metrics
        ax3 = fig.add_subplot(gs[1, 0])
        metrics = ['Top-K Accuracy', 'MRR', 'Precision@K']
        melted_df = pd.melt(
            self.results_df,
            id_vars=['Configuration', 'Type'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Score'
        )
        sns.boxplot(
            data=melted_df,
            x='Metric',
            y='Score',
            hue='Type',
            ax=ax3
        )
        ax3.set_title('Precision Metrics Distribution')
        
        # Plot chunk length variation
        ax4 = fig.add_subplot(gs[1, 1])
        sns.boxplot(
            data=self.results_df,
            x='Configuration',
            y='Chunk Length Std',
            hue='Type',
            ax=ax4
        )
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.set_title('Chunk Length Variation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_all_plots(self, output_dir: str = 'plots'):
        """Save all visualization plots to specified directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual metric comparisons
        metrics = ['Avg Similarity', 'Max Similarity', 'Top-K Accuracy', 
                  'MRR', 'Precision@K']
        for metric in metrics:
            self.plot_metric_comparison(metric)
            plt.savefig(
                f'{output_dir}/{metric.lower().replace("@", "_at_")}.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
        
        # Save chunk size distribution
        self.plot_chunk_size_distribution()
        plt.savefig(
            f'{output_dir}/chunk_size_distribution.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        # Save performance radar
        self.plot_performance_radar()
        plt.savefig(
            f'{output_dir}/performance_radar.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        # Save overview dashboard
        self.plot_overview_dashboard(f'{output_dir}/overview_dashboard.png')
        plt.close()

def create_comparison_plots(results_df: pd.DataFrame, output_dir: str = 'plots'):
    """Convenience function to create and save all comparison plots."""
    visualizer = ChunkingVisualizer(results_df)
    visualizer.save_all_plots(output_dir)