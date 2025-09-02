#!/usr/bin/env python3
"""
ADVANCED SIGNATURE GENERATOR - Creates realistic signature variations
Based on actual handwriting research and biometric studies
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import math

class AdvancedSignatureGenerator:
    """Advanced signature generator with realistic variations"""
    
    def __init__(self):
        # Signature templates based on real signature studies
        self.signature_templates = {
            0: self._cursive_template,
            1: self._initials_template,
            2: self._scrawl_template,
            3: self._mixed_template,
            4: self._ascending_template,
            5: self._descending_template,
            6: self._flourish_template,
            7: self._minimalist_template
        }
    
    def generate_signature(self, person_id, style='normal', width=200, height=100):
        """Generate signature with specified style"""
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Determine signature category and parameters
        signature_category = person_id % 8
        base_params = self._get_base_parameters(person_id)
        
        if style == 'normal':
            params = base_params
        elif style == 'variation':
            params = self._apply_realistic_variation(base_params)
        else:
            params = base_params
        
        # Generate signature using template
        template_func = self.signature_templates[signature_category]
        template_func(draw, params)
        
        return img
    
    def _get_base_parameters(self, person_id):
        """Get consistent base parameters for a person"""
        random.seed(person_id * 1000)  # Consistent seed
        
        params = {
            'person_id': person_id,
            'category': person_id % 8,
            'base_x': 20 + (person_id % 15),
            'base_y': 40 + (person_id % 12),
            'size_factor': 0.8 + (person_id % 6) * 0.1,  # 0.8 to 1.3
            'complexity': 0.6 + (person_id % 8) * 0.1,   # 0.6 to 1.3
            'stroke_width': 1 + (person_id % 3),         # 1 to 3
            'slant': -0.2 + (person_id % 5) * 0.1,       # -0.2 to 0.2
            'pressure': 0.8 + (person_id % 4) * 0.1,     # 0.8 to 1.1
            'speed': 0.9 + (person_id % 3) * 0.1,        # 0.9 to 1.1
        }
        
        random.seed()  # Reset seed
        return params
    
    def _apply_realistic_variation(self, base_params):
        """Apply realistic handwriting variations"""
        params = base_params.copy()
        
        # Natural handwriting variations (same person, different conditions)
        params['base_x'] += random.randint(-3, 3)      # Position drift
        params['base_y'] += random.randint(-2, 2)      # Position drift
        params['size_factor'] *= random.uniform(0.95, 1.05)  # ±5% size
        params['complexity'] *= random.uniform(0.92, 1.08)   # ±8% complexity
        params['stroke_width'] += random.uniform(-0.3, 0.3)  # Pressure variation
        params['slant'] += random.uniform(-0.05, 0.05)       # Slight slant change
        params['pressure'] *= random.uniform(0.95, 1.05)     # ±5% pressure
        params['speed'] *= random.uniform(0.98, 1.02)        # ±2% speed
        
        # Ensure reasonable bounds
        params['stroke_width'] = max(0.5, min(4.0, params['stroke_width']))
        params['size_factor'] = max(0.5, min(2.0, params['size_factor']))
        params['complexity'] = max(0.3, min(2.0, params['complexity']))
        
        return params
    
    def _cursive_template(self, draw, params):
        """Cursive signature template"""
        x, y = params['base_x'], params['base_y']
        size = params['size_factor']
        complexity = params['complexity']
        stroke = max(1, int(params['stroke_width']))
        
        # Create flowing cursive signature
        name_parts = ["John", "Smith"]
        x_pos = x
        
        for part_idx, part in enumerate(name_parts):
            for i, char in enumerate(part):
                char_width = int((8 + random.randint(-1, 1)) * size)
                
                # Connecting strokes between letters
                if i > 0:
                    connect_y = y + random.randint(-1, 1)
                    draw.line([(x_pos - 2, connect_y), (x_pos, y)], fill='black', width=1)
                
                # Letter shapes with natural variation
                height_var = int(6 * size * complexity) + random.randint(-1, 1)
                width_var = char_width + random.randint(-1, 1)
                
                # Draw letter as ellipse with slight variations
                draw.ellipse([(x_pos, y - height_var), (x_pos + width_var, y + height_var)], 
                           outline='black', width=stroke)
                
                # Add character-specific details
                if char.lower() in ['i', 'j', 't']:
                    # Add dots or crosses
                    dot_y = y - height_var - 3
                    draw.ellipse([(x_pos + width_var//2 - 1, dot_y - 1), 
                                (x_pos + width_var//2 + 1, dot_y + 1)], fill='black')
                
                x_pos += char_width + 1
            
            # Space between name parts
            x_pos += int(12 * size)
    
    def _initials_template(self, draw, params):
        """Stylized initials template"""
        x, y = params['base_x'], params['base_y']
        size = params['size_factor']
        stroke = max(1, int(params['stroke_width'] * params['complexity']))
        
        # Large decorative initials
        initial_size = int(25 * size)
        
        # First initial - decorative circle with line
        draw.ellipse([(x, y - 18), (x + initial_size, y + 7)], 
                    outline='black', width=stroke)
        draw.line([(x + initial_size//2, y - 18), (x + initial_size//2, y + 7)], 
                 fill='black', width=max(1, stroke - 1))
        
        # Second initial - vertical line with horizontal
        x2 = x + initial_size + 8
        draw.line([(x2, y - 18), (x2, y + 7)], fill='black', width=stroke)
        draw.line([(x2, y - 8), (x2 + initial_size//2, y - 8)], 
                 fill='black', width=max(1, stroke - 1))
        
        # Professional underline
        underline_length = int(90 * size)
        underline_y = y + 12 + random.randint(-1, 1)
        draw.line([(x, underline_y), (x + underline_length, underline_y)], 
                 fill='black', width=max(1, stroke - 1))
    
    def _scrawl_template(self, draw, params):
        """Illegible scrawl template"""
        x, y = params['base_x'], params['base_y']
        size = params['size_factor']
        complexity = params['complexity']
        speed = params['speed']
        
        # Create flowing scrawl pattern
        scrawl_length = int(85 * size)
        amplitude = int(12 * complexity)
        
        points = []
        for i in range(0, scrawl_length, max(1, int(4 / speed))):
            wave_x = x + i
            wave_y = y + amplitude * np.sin(i * 0.12 * complexity) * np.cos(i * 0.05)
            wave_y += random.randint(-2, 2)  # Natural hand tremor
            points.append((int(wave_x), int(wave_y)))
        
        # Draw connected lines with varying thickness
        for i in range(len(points) - 1):
            thickness = max(1, int(random.randint(1, 4) * params['pressure']))
            draw.line([points[i], points[i + 1]], fill='black', width=thickness)
    
    def _mixed_template(self, draw, params):
        """Mixed print-cursive template"""
        x, y = params['base_x'], params['base_y']
        size = params['size_factor']
        complexity = params['complexity']
        stroke = max(1, int(params['stroke_width']))
        
        # Print first name
        letters = "JOHN"
        x_pos = x
        letter_height = int(15 * size)
        
        for letter in letters:
            letter_width = int(8 * size)
            # Draw rectangular letter
            draw.rectangle([(x_pos, y - letter_height), (x_pos + letter_width, y + 3)], 
                          outline='black', width=stroke)
            
            # Add letter-specific details
            if letter in ['A', 'H', 'F', 'E']:
                # Add horizontal line
                mid_y = y - letter_height // 2
                draw.line([(x_pos + 1, mid_y), (x_pos + letter_width - 1, mid_y)], 
                         fill='black', width=max(1, stroke - 1))
            
            x_pos += int(12 * size)
        
        # Cursive last name
        x_pos += int(8 * size)
        for i in range(4):  # "Smith" approximation
            ellipse_width = int(10 * size * complexity)
            ellipse_height = int(4 * size)
            
            draw.ellipse([(x_pos, y - ellipse_height), (x_pos + ellipse_width, y + ellipse_height)], 
                       outline='black', width=stroke)
            
            # Connecting lines
            if i < 3:
                draw.line([(x_pos + ellipse_width - 1, y), (x_pos + ellipse_width + 3, y)], 
                         fill='black', width=1)
            
            x_pos += int(13 * size)
    
    def _ascending_template(self, draw, params):
        """Ascending signature template"""
        x, y = params['base_x'], params['base_y']
        size = params['size_factor']
        complexity = params['complexity']
        
        # Create ascending signature line
        points = []
        length = int(80 * size)
        
        for i in range(0, length, 3):
            point_x = x + i
            # Ascending trend with natural variation
            point_y = y - (i * 0.15) + 4 * np.sin(i * 0.08 * complexity)
            point_y += random.randint(-1, 1)
            points.append((int(point_x), int(point_y)))
        
        # Draw signature
        stroke = max(1, int(params['stroke_width']))
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill='black', width=stroke)
        
        # Add flourish at end
        if len(points) > 0:
            end_x, end_y = points[-1]
            draw.arc([(end_x - 8, end_y - 12), (end_x + 8, end_y + 4)], 
                    0, 180, fill='black', width=stroke)
    
    def _descending_template(self, draw, params):
        """Descending signature template"""
        x, y = params['base_x'], params['base_y']
        size = params['size_factor']
        complexity = params['complexity']
        
        # Create descending signature line
        points = []
        length = int(80 * size)
        
        for i in range(0, length, 3):
            point_x = x + i
            # Descending trend with natural variation
            point_y = y + (i * 0.12) + 3 * np.sin(i * 0.1 * complexity)
            point_y += random.randint(-1, 1)
            points.append((int(point_x), int(point_y)))
        
        # Draw signature
        stroke = max(1, int(params['stroke_width']))
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill='black', width=stroke)
    
    def _flourish_template(self, draw, params):
        """Elaborate flourish template"""
        x, y = params['base_x'], params['base_y']
        size = params['size_factor']
        complexity = params['complexity']
        stroke = max(1, int(params['stroke_width']))
        
        # Main signature line
        main_length = int(65 * size)
        draw.line([(x, y), (x + main_length, y - 3)], fill='black', width=stroke)
        
        # Decorative loops
        loop_size = int(12 * size * complexity)
        draw.ellipse([(x + 15, y - 18), (x + 15 + loop_size, y - 6)], 
                    outline='black', width=max(1, stroke - 1))
        draw.ellipse([(x + 35, y - 12), (x + 35 + loop_size, y + 2)], 
                    outline='black', width=max(1, stroke - 1))
        
        # Flourish tail
        tail_points = [
            (x + main_length, y - 3),
            (x + main_length + 8, y + 4),
            (x + main_length + 16, y - 8),
            (x + main_length + 20, y + 2)
        ]
        
        for i in range(len(tail_points) - 1):
            draw.line([tail_points[i], tail_points[i + 1]], fill='black', width=stroke)
    
    def _minimalist_template(self, draw, params):
        """Minimalist signature template"""
        x, y = params['base_x'], params['base_y']
        size = params['size_factor']
        stroke = max(1, int(params['stroke_width']))
        
        # Simple geometric signature
        line_length = int(55 * size)
        line_height = int(10 * size)
        
        # Horizontal line
        draw.line([(x, y), (x + line_length, y)], fill='black', width=stroke)
        
        # Vertical line
        draw.line([(x + line_length//2, y - line_height), 
                  (x + line_length//2, y + line_height)], 
                 fill='black', width=stroke)
        
        # Simple circle
        circle_size = int(4 * size)
        circle_x = x + line_length + 8
        draw.ellipse([(circle_x, y - circle_size), (circle_x + circle_size*2, y + circle_size)], 
                    outline='black', width=stroke)

# Global generator instance
generator = AdvancedSignatureGenerator()

def create_signature_image(text, width=200, height=100, style='normal', person_id=None):
    """Create signature using advanced generator"""
    if person_id is None:
        person_id = random.randint(1, 1000)
    
    return generator.generate_signature(person_id, style, width, height)