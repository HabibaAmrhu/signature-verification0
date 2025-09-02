#!/usr/bin/env python3
"""
PURE NUMPY SIGNATURE VERIFICATION ENSEMBLE - TARGET: 97%+ ACCURACY
Advanced ensemble using only numpy and PIL - no external dependencies
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import math

class PureNumpySignatureEnsemble:
    """Advanced ensemble using only numpy and PIL"""
    
    def __init__(self):
        self.algorithms = [
            self._enhanced_shape_algorithm,
            self._advanced_texture_algorithm,
            self._geometric_moments_algorithm,
            self._pixel_distribution_algorithm,
            self._structural_analysis_algorithm
        ]
        self.weights = [0.25, 0.20, 0.20, 0.20, 0.15]
    
    def verify_signatures(self, img1, img2):
        """Main verification using pure numpy ensemble"""
        try:
            # Advanced preprocessing
            proc1 = self._advanced_preprocessing(img1)
            proc2 = self._advanced_preprocessing(img2)
            
            # Pixel-level identity check first
            pixel_diff = np.mean(np.abs(proc1['primary'].astype(float) - proc2['primary'].astype(float)))
            if pixel_diff < 0.001:
                return 0.985  # Truly identical
            
            # Run ensemble algorithms
            scores = []
            confidences = []
            
            for algorithm in self.algorithms:
                try:
                    score, confidence = algorithm(proc1, proc2)
                    scores.append(score)
                    confidences.append(confidence)
                except Exception as e:
                    scores.append(0.5)
                    confidences.append(0.3)
            
            # Weighted ensemble
            weighted_score = 0.0
            total_weight = 0.0
            
            for score, confidence, base_weight in zip(scores, confidences, self.weights):
                adjusted_weight = base_weight * confidence
                weighted_score += score * adjusted_weight
                total_weight += adjusted_weight
            
            if total_weight == 0:
                return 0.5
            
            ensemble_score = weighted_score / total_weight
            
            # Advanced calibration
            final_score = self._advanced_calibration(ensemble_score, scores, confidences)
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            return 0.5
    
    def _advanced_preprocessing(self, img):
        """Advanced preprocessing with multiple variants"""
        gray = img.convert('L').resize((200, 200))
        arr = np.array(gray).astype(np.float32)
        
        # Multiple thresholding approaches
        variants = {}
        
        # 1. Conservative threshold
        variants['conservative'] = (arr < 200).astype(np.uint8)
        
        # 2. Adaptive threshold (pure numpy implementation)
        variants['adaptive'] = self._numpy_adaptive_threshold(arr)
        
        # 3. Otsu threshold
        variants['otsu'] = self._numpy_otsu_threshold(arr)
        
        # 4. Multi-level threshold
        variants['multilevel'] = self._numpy_multilevel_threshold(arr)
        
        # Choose best variant based on content
        best_variant = self._select_best_variant(variants)
        
        return {
            'original': arr,
            'variants': variants,
            'primary': best_variant,
            'metadata': self._calculate_preprocessing_metadata(best_variant)
        }
    
    def _numpy_adaptive_threshold(self, arr):
        """Pure numpy adaptive thresholding"""
        try:
            # Simple adaptive threshold using local mean
            kernel_size = 15
            half_kernel = kernel_size // 2
            
            result = np.zeros_like(arr, dtype=np.uint8)
            
            for i in range(half_kernel, arr.shape[0] - half_kernel):
                for j in range(half_kernel, arr.shape[1] - half_kernel):
                    local_region = arr[i-half_kernel:i+half_kernel+1, j-half_kernel:j+half_kernel+1]
                    local_mean = np.mean(local_region)
                    threshold = local_mean - 10  # Adaptive offset
                    
                    if arr[i, j] < threshold:
                        result[i, j] = 1
            
            return result
        except:
            return (arr < 200).astype(np.uint8)
    
    def _numpy_otsu_threshold(self, arr):
        """Pure numpy Otsu thresholding"""
        try:
            # Calculate histogram
            hist, bins = np.histogram(arr.flatten(), bins=256, range=(0, 256))
            
            # Normalize histogram
            hist = hist.astype(float)
            total_pixels = np.sum(hist)
            
            if total_pixels == 0:
                return (arr < 128).astype(np.uint8)
            
            # Calculate cumulative sums
            cum_sum = np.cumsum(hist)
            cum_mean = np.cumsum(hist * np.arange(256))
            
            # Find optimal threshold
            max_variance = 0
            optimal_threshold = 128
            
            for t in range(1, 255):
                w0 = cum_sum[t]
                w1 = total_pixels - w0
                
                if w0 == 0 or w1 == 0:
                    continue
                
                mu0 = cum_mean[t] / w0
                mu1 = (cum_mean[255] - cum_mean[t]) / w1
                
                variance = w0 * w1 * (mu0 - mu1) ** 2
                
                if variance > max_variance:
                    max_variance = variance
                    optimal_threshold = t
            
            return (arr < optimal_threshold).astype(np.uint8)
        except:
            return (arr < 200).astype(np.uint8)
    
    def _numpy_multilevel_threshold(self, arr):
        """Multi-level thresholding"""
        try:
            # Calculate multiple thresholds
            thresholds = [150, 180, 210]
            best_threshold = 200
            best_score = 0
            
            for threshold in thresholds:
                binary = (arr < threshold).astype(np.uint8)
                signature_pixels = np.sum(binary)
                
                # Score based on reasonable signature size
                if 100 < signature_pixels < 5000:
                    score = min(signature_pixels / 1000, 1.0)
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
            
            return (arr < best_threshold).astype(np.uint8)
        except:
            return (arr < 200).astype(np.uint8)
    
    def _select_best_variant(self, variants):
        """Select best preprocessing variant"""
        try:
            best_variant = variants['conservative']
            best_score = 0
            
            for name, variant in variants.items():
                signature_pixels = np.sum(variant)
                
                # Score based on reasonable signature characteristics
                if 50 < signature_pixels < 8000:
                    # Calculate compactness
                    coords = np.where(variant == 1)
                    if len(coords[0]) > 0:
                        height = coords[0].max() - coords[0].min() + 1
                        width = coords[1].max() - coords[1].min() + 1
                        compactness = signature_pixels / (height * width) if height * width > 0 else 0
                        
                        score = compactness * min(signature_pixels / 1000, 1.0)
                        
                        if score > best_score:
                            best_score = score
                            best_variant = variant
            
            return best_variant
        except:
            return variants['conservative']
    
    def _calculate_preprocessing_metadata(self, binary):
        """Calculate preprocessing quality metadata"""
        try:
            signature_pixels = np.sum(binary)
            
            if signature_pixels == 0:
                return {'quality': 0.0, 'signature_present': False}
            
            coords = np.where(binary == 1)
            height = coords[0].max() - coords[0].min() + 1
            width = coords[1].max() - coords[1].min() + 1
            
            # Quality metrics
            fill_ratio = signature_pixels / (height * width) if height * width > 0 else 0
            size_score = min(signature_pixels / 1000, 1.0)
            aspect_score = min(width / height, height / width) if height > 0 and width > 0 else 0
            
            quality = (fill_ratio + size_score + aspect_score) / 3
            
            return {
                'quality': quality,
                'signature_present': signature_pixels > 50,
                'pixel_count': signature_pixels,
                'dimensions': (width, height)
            }
        except:
            return {'quality': 0.0, 'signature_present': False}
    
    def _enhanced_shape_algorithm(self, proc1, proc2):
        """Enhanced shape-based algorithm"""
        try:
            features1 = self._extract_comprehensive_shape_features(proc1['primary'])
            features2 = self._extract_comprehensive_shape_features(proc2['primary'])
            
            if not features1['signature_present'] or not features2['signature_present']:
                if not features1['signature_present'] and not features2['signature_present']:
                    return 0.95, 0.9
                else:
                    return 0.15, 0.9
            
            similarity = self._calculate_shape_similarity(features1, features2)
            confidence = min(proc1['metadata']['quality'], proc2['metadata']['quality'])
            
            return similarity, max(0.5, confidence)
            
        except:
            return 0.5, 0.3
    
    def _extract_comprehensive_shape_features(self, binary):
        """Extract comprehensive shape features"""
        try:
            coords = np.where(binary == 1)
            if len(coords[0]) == 0:
                return {'signature_present': False}
            
            min_row, max_row = coords[0].min(), coords[0].max()
            min_col, max_col = coords[1].min(), coords[1].max()
            
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            signature_pixels = len(coords[0])
            
            features = {
                'signature_present': True,
                'aspect_ratio': width / height if height > 0 else 1.0,
                'fill_ratio': signature_pixels / (height * width) if height * width > 0 else 0.0,
                'bounding_box_ratio': (height * width) / (binary.shape[0] * binary.shape[1]),
                'center_x': np.mean(coords[1]) / binary.shape[1],
                'center_y': np.mean(coords[0]) / binary.shape[0],
                'pixel_count': signature_pixels,
                'width': width,
                'height': height,
                'compactness': self._calculate_compactness(binary, coords),
                'eccentricity': self._calculate_eccentricity_numpy(coords),
                'solidity': self._calculate_solidity(binary, coords),
                'extent': signature_pixels / (height * width) if height * width > 0 else 0,
                'orientation': self._calculate_orientation(coords)
            }
            
            return features
        except:
            return {'signature_present': False}
    
    def _calculate_compactness(self, binary, coords):
        """Calculate compactness using pure numpy"""
        try:
            signature_pixels = len(coords[0])
            perimeter = self._calculate_perimeter_numpy(binary)
            
            if perimeter == 0:
                return 0.0
            
            compactness = (4 * np.pi * signature_pixels) / (perimeter ** 2)
            return min(compactness, 1.0)
        except:
            return 0.5
    
    def _calculate_perimeter_numpy(self, binary):
        """Calculate perimeter using pure numpy"""
        try:
            perimeter = 0
            for i in range(1, binary.shape[0] - 1):
                for j in range(1, binary.shape[1] - 1):
                    if binary[i, j] == 1:
                        # Check 4-connectivity
                        neighbors = [
                            binary[i-1, j], binary[i+1, j],
                            binary[i, j-1], binary[i, j+1]
                        ]
                        if sum(neighbors) < 4:
                            perimeter += 1
            return max(perimeter, 1)
        except:
            return 1
    
    def _calculate_eccentricity_numpy(self, coords):
        """Calculate eccentricity using pure numpy"""
        try:
            if len(coords[0]) < 5:
                return 0.5
            
            # Calculate moments
            x = coords[1].astype(float)
            y = coords[0].astype(float)
            
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            # Central moments
            mu20 = np.mean((x - x_mean) ** 2)
            mu02 = np.mean((y - y_mean) ** 2)
            mu11 = np.mean((x - x_mean) * (y - y_mean))
            
            # Calculate eigenvalues
            trace = mu20 + mu02
            det = mu20 * mu02 - mu11 ** 2
            
            if det <= 0 or trace <= 0:
                return 0.5
            
            lambda1 = (trace + np.sqrt(trace ** 2 - 4 * det)) / 2
            lambda2 = (trace - np.sqrt(trace ** 2 - 4 * det)) / 2
            
            if lambda1 <= 0:
                return 0.5
            
            eccentricity = np.sqrt(1 - lambda2 / lambda1)
            return min(max(eccentricity, 0.0), 1.0)
        except:
            return 0.5
    
    def _calculate_solidity(self, binary, coords):
        """Calculate solidity (signature area / convex hull area)"""
        try:
            signature_pixels = len(coords[0])
            
            # Simple convex hull approximation using bounding box
            min_row, max_row = coords[0].min(), coords[0].max()
            min_col, max_col = coords[1].min(), coords[1].max()
            
            convex_area = (max_row - min_row + 1) * (max_col - min_col + 1)
            
            if convex_area == 0:
                return 0.0
            
            solidity = signature_pixels / convex_area
            return min(solidity, 1.0)
        except:
            return 0.5
    
    def _calculate_orientation(self, coords):
        """Calculate orientation angle"""
        try:
            if len(coords[0]) < 5:
                return 0.0
            
            x = coords[1].astype(float)
            y = coords[0].astype(float)
            
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            # Calculate orientation using second moments
            mu20 = np.mean((x - x_mean) ** 2)
            mu02 = np.mean((y - y_mean) ** 2)
            mu11 = np.mean((x - x_mean) * (y - y_mean))
            
            if mu20 == mu02:
                return 0.0
            
            orientation = 0.5 * np.arctan(2 * mu11 / (mu20 - mu02))
            return orientation / np.pi  # Normalize to [-0.5, 0.5]
        except:
            return 0.0
    
    def _calculate_shape_similarity(self, f1, f2):
        """Calculate shape similarity with advanced weighting"""
        weights = {
            'aspect_ratio': 0.15,
            'fill_ratio': 0.12,
            'bounding_box_ratio': 0.08,
            'center_x': 0.10,
            'center_y': 0.10,
            'compactness': 0.12,
            'eccentricity': 0.08,
            'solidity': 0.10,
            'extent': 0.08,
            'orientation': 0.07
        }
        
        total_similarity = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in f1 and feature in f2:
                val1, val2 = f1[feature], f2[feature]
                
                if feature in ['center_x', 'center_y']:
                    diff = abs(val1 - val2)
                    similarity = max(0.0, 1.0 - diff * 2.0)
                elif feature == 'orientation':
                    # Handle circular nature of orientation
                    diff = abs(val1 - val2)
                    diff = min(diff, 1.0 - diff)  # Circular distance
                    similarity = max(0.0, 1.0 - diff * 4.0)
                else:
                    max_val = max(abs(val1), abs(val2), 1e-8)
                    diff = abs(val1 - val2) / max_val
                    similarity = max(0.0, 1.0 - diff)
                
                total_similarity += similarity * weight
                total_weight += weight
        
        return total_similarity / total_weight if total_weight > 0 else 0.0
    
    def _advanced_texture_algorithm(self, proc1, proc2):
        """Advanced texture analysis"""
        try:
            texture1 = self._extract_texture_features_numpy(proc1['primary'])
            texture2 = self._extract_texture_features_numpy(proc2['primary'])
            
            if len(texture1) == 0 or len(texture2) == 0:
                return 0.5, 0.3
            
            similarity = self._calculate_texture_similarity(texture1, texture2)
            confidence = 0.8 if len(texture1) > 20 and len(texture2) > 20 else 0.5
            
            return similarity, confidence
        except:
            return 0.5, 0.3
    
    def _extract_texture_features_numpy(self, binary):
        """Extract texture features using pure numpy"""
        try:
            features = []
            
            # Simple Local Binary Pattern implementation
            for i in range(1, binary.shape[0] - 1):
                for j in range(1, binary.shape[1] - 1):
                    if binary[i, j] > 0:
                        center = binary[i, j]
                        neighbors = [
                            binary[i-1, j-1], binary[i-1, j], binary[i-1, j+1],
                            binary[i, j+1], binary[i+1, j+1], binary[i+1, j],
                            binary[i+1, j-1], binary[i, j-1]
                        ]
                        
                        # Calculate LBP value
                        lbp_value = 0
                        for k, neighbor in enumerate(neighbors):
                            if neighbor >= center:
                                lbp_value += 2 ** k
                        
                        features.append(lbp_value % 256)  # Keep in byte range
            
            return features
        except:
            return []
    
    def _calculate_texture_similarity(self, texture1, texture2):
        """Calculate texture similarity"""
        try:
            # Create histograms
            hist1, _ = np.histogram(texture1, bins=32, range=(0, 256))
            hist2, _ = np.histogram(texture2, bins=32, range=(0, 256))
            
            # Normalize
            hist1 = hist1.astype(float) / (np.sum(hist1) + 1e-8)
            hist2 = hist2.astype(float) / (np.sum(hist2) + 1e-8)
            
            # Calculate correlation
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            
            if np.isnan(correlation):
                return 0.5
            
            return (correlation + 1) / 2
        except:
            return 0.5
    
    def _geometric_moments_algorithm(self, proc1, proc2):
        """Geometric moments algorithm"""
        try:
            moments1 = self._calculate_geometric_moments(proc1['primary'])
            moments2 = self._calculate_geometric_moments(proc2['primary'])
            
            if moments1 is None or moments2 is None:
                return 0.5, 0.3
            
            similarity = self._compare_moments(moments1, moments2)
            confidence = 0.7
            
            return similarity, confidence
        except:
            return 0.5, 0.3
    
    def _calculate_geometric_moments(self, binary):
        """Calculate geometric moments"""
        try:
            coords = np.where(binary == 1)
            if len(coords[0]) == 0:
                return None
            
            x = coords[1].astype(float)
            y = coords[0].astype(float)
            
            # Calculate central moments
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            moments = {
                'm00': len(x),  # Area
                'm10': np.sum(x),
                'm01': np.sum(y),
                'm20': np.sum((x - x_mean) ** 2),
                'm02': np.sum((y - y_mean) ** 2),
                'm11': np.sum((x - x_mean) * (y - y_mean)),
                'm30': np.sum((x - x_mean) ** 3),
                'm03': np.sum((y - y_mean) ** 3),
                'm21': np.sum((x - x_mean) ** 2 * (y - y_mean)),
                'm12': np.sum((x - x_mean) * (y - y_mean) ** 2)
            }
            
            return moments
        except:
            return None
    
    def _compare_moments(self, moments1, moments2):
        """Compare geometric moments"""
        try:
            # Normalize moments
            keys = ['m20', 'm02', 'm11', 'm30', 'm03', 'm21', 'm12']
            
            similarities = []
            for key in keys:
                if key in moments1 and key in moments2:
                    val1 = moments1[key] / (moments1['m00'] + 1e-8)
                    val2 = moments2[key] / (moments2['m00'] + 1e-8)
                    
                    max_val = max(abs(val1), abs(val2), 1e-8)
                    diff = abs(val1 - val2) / max_val
                    similarity = max(0.0, 1.0 - diff)
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.5
        except:
            return 0.5
    
    def _pixel_distribution_algorithm(self, proc1, proc2):
        """Pixel distribution algorithm"""
        try:
            dist1 = self._calculate_pixel_distribution(proc1['primary'])
            dist2 = self._calculate_pixel_distribution(proc2['primary'])
            
            similarity = self._compare_distributions(dist1, dist2)
            confidence = 0.6
            
            return similarity, confidence
        except:
            return 0.5, 0.3
    
    def _calculate_pixel_distribution(self, binary):
        """Calculate pixel distribution features"""
        try:
            coords = np.where(binary == 1)
            if len(coords[0]) == 0:
                return None
            
            # Divide image into grid and calculate density
            grid_size = 8
            h_step = binary.shape[0] // grid_size
            w_step = binary.shape[1] // grid_size
            
            distribution = []
            for i in range(grid_size):
                for j in range(grid_size):
                    y_start = i * h_step
                    y_end = min((i + 1) * h_step, binary.shape[0])
                    x_start = j * w_step
                    x_end = min((j + 1) * w_step, binary.shape[1])
                    
                    patch = binary[y_start:y_end, x_start:x_end]
                    density = np.sum(patch) / (patch.size + 1e-8)
                    distribution.append(density)
            
            return np.array(distribution)
        except:
            return None
    
    def _compare_distributions(self, dist1, dist2):
        """Compare pixel distributions"""
        try:
            if dist1 is None or dist2 is None:
                return 0.5
            
            # Calculate correlation
            correlation = np.corrcoef(dist1, dist2)[0, 1]
            
            if np.isnan(correlation):
                return 0.5
            
            return (correlation + 1) / 2
        except:
            return 0.5
    
    def _structural_analysis_algorithm(self, proc1, proc2):
        """Structural analysis algorithm"""
        try:
            struct1 = self._analyze_structure(proc1['primary'])
            struct2 = self._analyze_structure(proc2['primary'])
            
            similarity = self._compare_structures(struct1, struct2)
            confidence = 0.7
            
            return similarity, confidence
        except:
            return 0.5, 0.3
    
    def _analyze_structure(self, binary):
        """Analyze signature structure"""
        try:
            coords = np.where(binary == 1)
            if len(coords[0]) == 0:
                return None
            
            # Calculate structural features
            structure = {
                'connected_components': self._count_connected_components(binary),
                'skeleton_length': self._estimate_skeleton_length(binary, coords),
                'endpoints': self._count_endpoints(binary),
                'branching_points': self._count_branching_points(binary),
                'stroke_width_variation': self._calculate_stroke_width_variation(binary, coords)
            }
            
            return structure
        except:
            return None
    
    def _count_connected_components(self, binary):
        """Count connected components using flood fill"""
        try:
            visited = np.zeros_like(binary, dtype=bool)
            components = 0
            
            for i in range(binary.shape[0]):
                for j in range(binary.shape[1]):
                    if binary[i, j] == 1 and not visited[i, j]:
                        # Flood fill
                        stack = [(i, j)]
                        while stack:
                            y, x = stack.pop()
                            if (0 <= y < binary.shape[0] and 0 <= x < binary.shape[1] and
                                binary[y, x] == 1 and not visited[y, x]):
                                visited[y, x] = True
                                # Add 8-connected neighbors
                                for dy in [-1, 0, 1]:
                                    for dx in [-1, 0, 1]:
                                        if dy != 0 or dx != 0:
                                            stack.append((y + dy, x + dx))
                        components += 1
            
            return components
        except:
            return 1
    
    def _estimate_skeleton_length(self, binary, coords):
        """Estimate skeleton length"""
        try:
            # Simple skeleton length estimation
            if len(coords[0]) == 0:
                return 0
            
            # Use perimeter as rough skeleton estimate
            perimeter = self._calculate_perimeter_numpy(binary)
            return perimeter * 0.7  # Rough conversion factor
        except:
            return 0
    
    def _count_endpoints(self, binary):
        """Count endpoints in signature"""
        try:
            endpoints = 0
            for i in range(1, binary.shape[0] - 1):
                for j in range(1, binary.shape[1] - 1):
                    if binary[i, j] == 1:
                        # Count 8-connected neighbors
                        neighbors = 0
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di != 0 or dj != 0:
                                    if binary[i + di, j + dj] == 1:
                                        neighbors += 1
                        
                        # Endpoint has exactly 1 neighbor
                        if neighbors == 1:
                            endpoints += 1
            
            return endpoints
        except:
            return 0
    
    def _count_branching_points(self, binary):
        """Count branching points"""
        try:
            branching_points = 0
            for i in range(1, binary.shape[0] - 1):
                for j in range(1, binary.shape[1] - 1):
                    if binary[i, j] == 1:
                        # Count 8-connected neighbors
                        neighbors = 0
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di != 0 or dj != 0:
                                    if binary[i + di, j + dj] == 1:
                                        neighbors += 1
                        
                        # Branching point has 3+ neighbors
                        if neighbors >= 3:
                            branching_points += 1
            
            return branching_points
        except:
            return 0
    
    def _calculate_stroke_width_variation(self, binary, coords):
        """Calculate stroke width variation"""
        try:
            if len(coords[0]) == 0:
                return 0
            
            # Simple stroke width estimation
            signature_pixels = len(coords[0])
            perimeter = self._calculate_perimeter_numpy(binary)
            
            if perimeter == 0:
                return 0
            
            # Average stroke width approximation
            avg_stroke_width = signature_pixels / perimeter
            
            # Calculate variation (simplified)
            return min(avg_stroke_width, 10.0)  # Cap at reasonable value
        except:
            return 0
    
    def _compare_structures(self, struct1, struct2):
        """Compare structural features"""
        try:
            if struct1 is None or struct2 is None:
                return 0.5
            
            weights = {
                'connected_components': 0.3,
                'skeleton_length': 0.2,
                'endpoints': 0.2,
                'branching_points': 0.2,
                'stroke_width_variation': 0.1
            }
            
            total_similarity = 0.0
            total_weight = 0.0
            
            for feature, weight in weights.items():
                if feature in struct1 and feature in struct2:
                    val1, val2 = struct1[feature], struct2[feature]
                    
                    if feature == 'connected_components':
                        # Exact match is best for components
                        if val1 == val2:
                            similarity = 1.0
                        else:
                            max_val = max(val1, val2, 1)
                            similarity = 1.0 - abs(val1 - val2) / max_val
                    else:
                        # Relative similarity for other features
                        max_val = max(abs(val1), abs(val2), 1e-8)
                        diff = abs(val1 - val2) / max_val
                        similarity = max(0.0, 1.0 - diff)
                    
                    total_similarity += similarity * weight
                    total_weight += weight
            
            return total_similarity / total_weight if total_weight > 0 else 0.5
        except:
            return 0.5
    
    def _advanced_calibration(self, raw_score, individual_scores, confidences):
        """Advanced ensemble calibration"""
        try:
            # Calculate algorithm agreement
            if len(individual_scores) > 1:
                agreement = 1.0 - np.std(individual_scores)
            else:
                agreement = 0.8
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.5
            
            # Adjust score based on agreement and confidence
            if agreement > 0.85 and avg_confidence > 0.7:
                # High agreement and confidence
                calibrated = raw_score
            elif agreement > 0.7:
                # Moderate agreement
                calibrated = raw_score * 0.95 + 0.025
            else:
                # Low agreement - be conservative
                calibrated = raw_score * 0.9 + 0.05
            
            # Final calibration mapping
            if calibrated > 0.95:
                final_score = 0.95 + (calibrated - 0.95) * 1.0
            elif calibrated > 0.85:
                final_score = 0.78 + (calibrated - 0.85) * 1.7
            elif calibrated > 0.70:
                final_score = 0.55 + (calibrated - 0.70) * 1.53
            elif calibrated > 0.50:
                final_score = 0.25 + (calibrated - 0.50) * 1.5
            else:
                final_score = calibrated * 0.5
            
            return max(0.0, min(1.0, final_score))
        except:
            return raw_score

# Global ensemble instance
ensemble = PureNumpySignatureEnsemble()

def create_demo_prediction(img1, img2):
    """Main prediction function using pure numpy ensemble"""
    return ensemble.verify_signatures(img1, img2)