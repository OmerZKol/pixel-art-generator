import os
import numpy as np
from PIL import Image

def create_sample_sprites():
    """Create some sample 16x16 pixel art sprites for testing"""
    os.makedirs('sprites', exist_ok=True)
    
    # More varied patterns for diffusion training
    patterns = []
    pattern_names = []
    # Geometric patterns
    for i in range(10):
        pattern = np.zeros((16, 16))
        # Random geometric shapes
        if i % 3 == 0:  # Circles
            center_x, center_y = np.random.randint(4, 12, 2)
            radius = np.random.randint(2, 5)
            y, x = np.ogrid[:16, :16]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            pattern[mask] = 1
            pattern_names.append("circle")
        elif i % 3 == 1:  # Rectangles
            x1, y1 = np.random.randint(0, 8, 2)
            x2, y2 = np.random.randint(8, 16, 2)
            pattern[y1:y2, x1:x2] = 1
            pattern_names.append("rectangle")
        else:  # Lines
            if np.random.random() > 0.5:  # Horizontal
                y = np.random.randint(2, 14)
                pattern[y, :] = 1
            else:  # Vertical
                x = np.random.randint(2, 14)
                pattern[:, x] = 1
            pattern_names.append("line")
        patterns.append(pattern)
    
    # Character-like patterns
    char_patterns = [
        # Simple smiley
        np.array([
            [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
            [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
            [1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0],
            [1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,0],
            [1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0],
            [1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
            [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        ])
    ]
    patterns.extend(char_patterns)
    pattern_names.append("face")
    
    colours = [
        (255, 50, 50),    # Red
        (50, 255, 50),    # Green
        (50, 50, 255),    # Blue
        (255, 255, 50),   # Yellow
        (255, 50, 255),   # Magenta
        (50, 255, 255),   # Cyan
        (255, 150, 50),   # Orange
        (150, 50, 255),   # Purple
    ]
    colour_names = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
    
    # Generate sprites
    sprite_count = 0
    for index, pattern in enumerate(patterns):
        pattern_name = pattern_names[index]
        for c_index, colour in enumerate(colours):
            img = np.zeros((16, 16, 3), dtype=np.uint8)
            colour_name = colour_names[c_index]
            
            # Main color
            for i in range(16):
                for j in range(16):
                    if pattern[i, j]:
                        img[i, j] = colour
            
            # Add background color (subtle)
            bg_color = [c // 8 for c in colour]  # Very dark version of main color
            for i in range(16):
                for j in range(16):
                    if not pattern[i, j]:
                        img[i, j] = bg_color
            
            # Add some texture/noise
            noise = np.random.randint(-10, 10, (16, 16, 3))
            img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
            
            # Save sprite
            Image.fromarray(img).save(f'sprites/sprite_{pattern_name}_{colour_name}_{sprite_count:03d}.png')
            sprite_count += 1
    
    print(f"Created {sprite_count} sample sprites")

# create_sample_sprites()