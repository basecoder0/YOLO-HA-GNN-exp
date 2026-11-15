import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from typing import List, Dict, Optional
import cv2


class Visualizer:
    """visualizationtoolclass"""

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['pedestrian', 'vehicle', 'bicycle']
        self.colors = {
            'pedestrian': 'red',
            'vehicle': 'blue',
            'bicycle': 'green'
        }
        self.risk_colors = {
            'low': 'green',
            'medium': 'yellow',
            'high': 'orange',
            'critical': 'red'
        }

    def plot_detections(self,
                        image: np.ndarray,
                        detections: List[Dict],
                        title: str = "Detections",
                        show_confidence: bool = True,
                        show_risk: bool = False) -> plt.Figure:
        """drawdetectionresult"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        for det in detections:
            bbox = det.get('bbox', [])
            class_id = det.get('class_id', 0)
            confidence = det.get('confidence', 0.0)
            risk_score = det.get('preliminary_risk', 0.0)

            if len(bbox) == 4:
                # drawboundarybox
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                color = self.colors.get(class_name, 'white')

                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)

                # addlabel
                label_parts = [class_name]
                if show_confidence:
                    label_parts.append(f'conf: {confidence:.2f}')
                if show_risk and risk_score > 0:
                    label_parts.append(f'risk: {risk_score:.2f}')

                label = ' '.join(label_parts)
                ax.text(bbox[0], bbox[1] - 5, label,
                        color=color, fontsize=8, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))

        ax.set_title(title)
        ax.axis('off')

        return fig

    def plot_risk_assessment(self,
                             image: np.ndarray,
                             risk_results: Dict,
                             title: str = "Risk Assessment") -> plt.Figure:
        """drawriskevaluationresult"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左侧：detectionresult
        ax1.imshow(image)
        ax1.set_title("Detections with Risk Scores")

        # 右侧：riskanalysis
        ax2.axis('off')
        ax2.set_title("Risk Analysis")

        # drawtargetrisk
        target_risks = risk_results.get('target_risks', [])
        y_pos = 0.9
        for target_risk in target_risks:
            risk_level = self._get_risk_level(target_risk['combined_risk'])
            color = self.risk_colors.get(risk_level, 'gray')

            risk_text = (f"Target {target_risk['target_id']} ({target_risk['class_name']}): "
                         f"{target_risk['combined_risk']:.3f}")
            ax2.text(0.1, y_pos, risk_text, color=color, fontsize=10, weight='bold',
                     transform=ax2.transAxes)
            y_pos -= 0.08

            # at/in左侧imageabovedrawriskbox
            bbox = target_risk.get('bbox', [])
            if len(bbox) == 4:
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    linewidth=3, edgecolor=color, facecolor='none'
                )
                ax1.add_patch(rect)

                # addriskscore
                ax1.text(bbox[0], bbox[1] - 8, f"Risk: {target_risk['combined_risk']:.2f}",
                         color=color, fontsize=8, weight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))

        # drawconflict pair
        conflict_pairs = risk_results.get('conflict_pairs', [])
        y_pos -= 0.05
        ax2.text(0.1, y_pos, "Conflict Pairs:", fontsize=10, weight='bold', transform=ax2.transAxes)
        y_pos -= 0.05

        for conflict in conflict_pairs:
            risk_level = self._get_risk_level(conflict['conflict_risk'])
            color = self.risk_colors.get(risk_level, 'gray')

            conflict_text = (f"{conflict['pair_type']}: {conflict['conflict_risk']:.3f} "
                             f"(dist: {conflict['distance']:.1f}m)")
            ax2.text(0.15, y_pos, conflict_text, color=color, fontsize=9, transform=ax2.transAxes)
            y_pos -= 0.05

        # totalbodyrisk
        overall_risk = risk_results.get('overall_risk', 0.0)
        overall_level = self._get_risk_level(overall_risk)
        overall_color = self.risk_colors.get(overall_level, 'gray')

        ax2.text(0.1, 0.05, f"Overall Risk: {overall_risk:.3f}",
                 color=overall_color, fontsize=12, weight='bold', transform=ax2.transAxes)

        # predict/prealertinformation
        warnings = risk_results.get('warnings', [])
        if warnings:
            ax2.text(0.1, 0.02, f"Warning: {warnings[0]}",
                     color='red', fontsize=10, weight='bold', transform=ax2.transAxes)

        plt.tight_layout()
        return fig

    def _get_risk_level(self, risk_score: float) -> str:
        """rootaccordingriskscoregetriskwaitlevel"""
        if risk_score > 0.8:
            return 'critical'
        elif risk_score > 0.6:
            return 'high'
        elif risk_score > 0.4:
            return 'medium'
        else:
            return 'low'

    def plot_attention_weights(self, attention_weights: Dict) -> plt.Figure:
        """drawnoteabilityweight"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # emptyspacenoteabilityweight
        if 'spatial_weights' in attention_weights:
            spatial_weights = attention_weights['spatial_weights']
            if spatial_weights.numel() > 0:
                im1 = axes[0, 0].imshow(spatial_weights[0, 0].cpu().detach().numpy(), cmap='hot')
                axes[0, 0].set_title('Spatial Attention Weights')
                plt.colorbar(im1, ax=axes[0, 0])

        # channelnoteabilityweight（RGB）
        if 'channel_weights_rgb' in attention_weights:
            channel_rgb = attention_weights['channel_weights_rgb']
            if channel_rgb.numel() > 0:
                axes[0, 1].bar(range(len(channel_rgb[0])), channel_rgb[0].cpu().detach().numpy())
                axes[0, 1].set_title('Channel Attention Weights (RGB)')
                axes[0, 1].set_xlabel('Channel')
                axes[0, 1].set_ylabel('Weight')

        # channelnoteabilityweight（Thermal）
        if 'channel_weights_thermal' in attention_weights:
            channel_thermal = attention_weights['channel_weights_thermal']
            if channel_thermal.numel() > 0:
                axes[1, 0].bar(range(len(channel_thermal[0])), channel_thermal[0].cpu().detach().numpy())
                axes[1, 0].set_title('Channel Attention Weights (Thermal)')
                axes[1, 0].set_xlabel('Channel')
                axes[1, 0].set_ylabel('Weight')

        # traininglosscurveline
        axes[1, 1].axis('off')

        plt.tight_layout()
        return fig

    def create_comparison_plot(self,
                               original_image: np.ndarray,
                               stage1_detections: List[Dict],
                               enhanced_detections: List[Dict],
                               risk_assessment: Dict) -> plt.Figure:
        """createcomparisonimage：showthreeastageofresult"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # originalimage
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        # firststagedetectionresult
        axes[0, 1].imshow(original_image)
        for det in stage1_detections:
            bbox = det.get('bbox', [])
            if len(bbox) == 4:
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    linewidth=2, edgecolor='blue', facecolor='none'
                )
                axes[0, 1].add_patch(rect)
                axes[0, 1].text(bbox[0], bbox[1] - 5,
                                f"conf: {det.get('confidence', 0):.2f}",
                                color='blue', fontsize=8, weight='bold')
        axes[0, 1].set_title("Stage 1: YOLO Detections")
        axes[0, 1].axis('off')

        # secondstageenhanceresult
        axes[1, 0].imshow(original_image)
        for det in enhanced_detections:
            bbox = det.get('enhanced_bbox', det.get('bbox', []))
            if len(bbox) == 4:
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                axes[1, 0].add_patch(rect)
                risk_text = f"risk: {det.get('preliminary_risk', 0):.2f}"
                axes[1, 0].text(bbox[0], bbox[1] - 5, risk_text,
                                color='green', fontsize=8, weight='bold')
        axes[1, 0].set_title("Stage 2: GNN Enhanced")
        axes[1, 0].axis('off')

        # thethreestageriskevaluation
        self._plot_risk_summary(axes[1, 1], risk_assessment)
        axes[1, 1].set_title("Stage 3: Risk Assessment")

        plt.tight_layout()
        return fig

    def _plot_risk_summary(self, ax, risk_assessment: Dict):
        """drawrisktotal结"""
        ax.axis('off')

        overall_risk = risk_assessment.get('overall_risk', 0.0)
        risk_level = self._get_risk_level(overall_risk)
        color = self.risk_colors.get(risk_level, 'gray')

        # totalbodyrisk
        ax.text(0.1, 0.9, f"Overall Risk: {overall_risk:.3f}",
                color=color, fontsize=14, weight='bold', transform=ax.transAxes)

        # highrisktarget
        high_risk_targets = [t for t in risk_assessment.get('target_risks', [])
                             if t['combined_risk'] > 0.6]

        ax.text(0.1, 0.8, f"High Risk Targets: {len(high_risk_targets)}",
                fontsize=12, transform=ax.transAxes)

        # conflict paircountamount
        conflict_pairs = risk_assessment.get('conflict_pairs', [])
        high_risk_conflicts = [c for c in conflict_pairs if c['conflict_risk'] > 0.6]

        ax.text(0.1, 0.7, f"High Risk Conflicts: {len(high_risk_conflicts)}",
                fontsize=12, transform=ax.transAxes)

        # predict/prealertinformation
        warnings = risk_assessment.get('warnings', [])
        if warnings:
            ax.text(0.1, 0.6, f"Warnings: {len(warnings)}",
                    color='red', fontsize=12, weight='bold', transform=ax.transAxes)
            for i, warning in enumerate(warnings[:3]):  # showfirst3awarning
                ax.text(0.15, 0.5 - i * 0.08, f"• {warning}",
                        color='red', fontsize=10, transform=ax.transAxes)