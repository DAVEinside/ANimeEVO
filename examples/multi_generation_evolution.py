"""
Multi-generation evolution example.

This script demonstrates evolving characters over multiple generations
with population dynamics and fitness selection.
"""

import os
import sys
import time
import logging
import random
from pathlib import Path

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from core.attributes.character_attributes import CharacterAttributes
from core.diffusion.anime_pipeline import AnimePipeline
from core.evolution.evolution_engine import EvolutionEngine
from multimodal.image_generator import ImageGenerator
from utils.image_processing import create_image_grid, add_overlay_text
from utils.data_handling import ensure_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomFitnessEvaluator:
    """
    Custom fitness evaluator for character evolution.
    """
    
    def __init__(self, target_traits=None):
        """
        Initialize the fitness evaluator.
        
        Args:
            target_traits: Dictionary of target traits for evolution
        """
        self.target_traits = target_traits or {
            'hair_color': ['blue', 'purple'],
            'eye_color': ['green', 'gold'],
            'personality': ['brave', 'determined', 'cheerful'],
            'anime_style': 'shonen'
        }
        
    def evaluate(self, character: CharacterAttributes) -> float:
        """
        Evaluate fitness of a character.
        
        Args:
            character: Character to evaluate
            
        Returns:
            Fitness score (0-1)
        """
        score = 0.0
        total_weight = 0.0
        
        # Check hair color
        if hasattr(character, 'hair_color') and character.hair_color:
            weight = 0.2
            total_weight += weight
            if character.hair_color in self.target_traits.get('hair_color', []):
                score += weight
                
        # Check eye color
        if hasattr(character, 'eye_color') and character.eye_color:
            weight = 0.2
            total_weight += weight
            if character.eye_color in self.target_traits.get('eye_color', []):
                score += weight
                
        # Check personality traits
        if hasattr(character, 'personality') and character.personality:
            weight = 0.3
            total_weight += weight
            target_personality = self.target_traits.get('personality', [])
            matches = sum(1 for trait in character.personality if trait in target_personality)
            if matches > 0 and len(target_personality) > 0:
                score += weight * (matches / len(target_personality))
                
        # Check anime style
        if hasattr(character, 'anime_style') and character.anime_style:
            weight = 0.3
            total_weight += weight
            if character.anime_style == self.target_traits.get('anime_style'):
                score += weight
                
        # Normalize score
        if total_weight > 0:
            return score / total_weight
        return 0.0

def run_multi_generation_evolution(
    generations: int = 5,
    population_size: int = 8,
    output_dir: str = "outputs/multi_generation",
    create_visualizations: bool = True
):
    """
    Run multi-generation evolution.
    
    Args:
        generations: Number of generations to evolve
        population_size: Size of the population
        output_dir: Directory to save outputs
        create_visualizations: Whether to create visualizations
    """
    logger.info(f"Running multi-generation evolution: {generations} generations, {population_size} population size")
    
    # Create output directory
    ensure_directory(output_dir)
    
    # Initialize components
    pipeline = AnimePipeline()
    evolution_engine = EvolutionEngine()
    image_generator = ImageGenerator(pipeline=pipeline)
    
    # Set evolution parameters
    evolution_engine.population_size = population_size
    evolution_engine.mutation_rate = 0.3
    evolution_engine.crossover_rate = 0.7
    evolution_engine.elite_size = 2
    
    # Create fitness evaluator
    fitness_evaluator = CustomFitnessEvaluator()
    
    # Create initial population
    logger.info("Creating initial population...")
    
    # Start with different anime styles for diversity
    initial_population = []
    
    styles = ['shonen', 'shojo', 'seinen', 'isekai', 'mecha', 'chibi', 'modern_anime']
    for i in range(population_size):
        style = styles[i % len(styles)]
        character = CharacterAttributes(
            name=f"Gen0_Char{i+1}",
            gender=random.choice(['male', 'female']),
            anime_style=style
        )
        initial_population.append(character)
    
    # Generate images for initial population
    logger.info("Generating images for initial population...")
    for i, character in enumerate(initial_population):
        images, _ = image_generator.generate_images(
            attributes=character,
            num_samples=1
        )
        
        if images:
            # Save the image
            image_path = os.path.join(output_dir, f"gen0_char{i+1}.png")
            images[0].save(image_path)
            logger.info(f"Saved image: {image_path}")
    
    # Current population
    current_population = initial_population
    
    # Track best fitness per generation
    best_fitness_history = []
    avg_fitness_history = []
    
    # Evolve for multiple generations
    for gen in range(1, generations + 1):
        logger.info(f"Evolving generation {gen}...")
        
        # Calculate fitness for each character
        for character in current_population:
            character.fitness_score = fitness_evaluator.evaluate(character)
            
        # Get fitness stats for this generation
        fitness_values = [c.fitness_score for c in current_population]
        best_fitness = max(fitness_values)
        avg_fitness = sum(fitness_values) / len(fitness_values)
        
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        logger.info(f"Generation {gen-1} stats: Best fitness: {best_fitness:.4f}, Avg fitness: {avg_fitness:.4f}")
        
        # Sort by fitness for visualization
        current_population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Create generation visualization
        if create_visualizations:
            create_generation_visualization(
                population=current_population,
                generation=gen-1,
                output_dir=output_dir,
                image_generator=image_generator
            )
        
        # Evolve to next generation
        current_population = evolution_engine.evolve_generation(current_population)
        
        # Generate images for new generation
        logger.info(f"Generating images for generation {gen}...")
        for i, character in enumerate(current_population):
            images, _ = image_generator.generate_images(
                attributes=character,
                num_samples=1
            )
            
            if images:
                # Save the image
                image_path = os.path.join(output_dir, f"gen{gen}_char{i+1}.png")
                images[0].save(image_path)
                logger.info(f"Saved image: {image_path}")
    
    # Calculate final fitness
    for character in current_population:
        character.fitness_score = fitness_evaluator.evaluate(character)
        
    # Get fitness stats for the final generation
    fitness_values = [c.fitness_score for c in current_population]
    best_fitness = max(fitness_values)
    avg_fitness = sum(fitness_values) / len(fitness_values)
    
    best_fitness_history.append(best_fitness)
    avg_fitness_history.append(avg_fitness)
    
    logger.info(f"Final generation stats: Best fitness: {best_fitness:.4f}, Avg fitness: {avg_fitness:.4f}")
    
    # Sort by fitness for final visualization
    current_population.sort(key=lambda x: x.fitness_score, reverse=True)
    
    # Create final generation visualization
    if create_visualizations:
        create_generation_visualization(
            population=current_population,
            generation=generations,
            output_dir=output_dir,
            image_generator=image_generator
        )
        
    # Create fitness history plot
    if create_visualizations:
        create_fitness_plot(
            best_fitness_history=best_fitness_history,
            avg_fitness_history=avg_fitness_history,
            output_dir=output_dir
        )
    
    logger.info(f"Multi-generation evolution complete. Results saved to {output_dir}")

def create_generation_visualization(
    population: list,
    generation: int,
    output_dir: str,
    image_generator: ImageGenerator
):
    """
    Create a visualization of a generation.
    
    Args:
        population: List of characters in the generation
        generation: Generation number
        output_dir: Directory to save output
        image_generator: Image generator instance
    """
    logger.info(f"Creating visualization for generation {generation}...")
    
    # Generate images for all characters
    images = []
    for i, character in enumerate(population):
        # Try to load existing image first
        image_path = os.path.join(output_dir, f"gen{generation}_char{i+1}.png")
        if os.path.exists(image_path):
            try:
                from PIL import Image
                img = Image.open(image_path)
                images.append(img)
                continue
            except:
                pass
        
        # Generate image if not already saved
        imgs, _ = image_generator.generate_images(
            attributes=character,
            num_samples=1
        )
        
        if imgs:
            images.append(imgs[0])
    
    if not images:
        logger.warning(f"No images available for generation {generation}")
        return
    
    # Add labels to images
    labeled_images = []
    for i, (img, character) in enumerate(zip(images, population)):
        label_text = f"Fitness: {character.fitness_score:.4f}\n{character.name}"
        labeled_img = add_overlay_text(img, label_text, position='bottom')
        labeled_images.append(labeled_img)
    
    # Create grid
    grid = create_image_grid(labeled_images, rows=2)
    
    # Add generation title
    from PIL import Image, ImageDraw, ImageFont
    
    # Create space for title
    height_with_title = grid.height + 60
    final_img = Image.new('RGB', (grid.width, height_with_title), color=(255, 255, 255))
    final_img.paste(grid, (0, 60))
    
    # Add title
    draw = ImageDraw.Draw(final_img)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    title = f"Generation {generation}"
    draw.text((final_img.width // 2, 30), title, fill=(0, 0, 0), font=font, anchor="mm")
    
    # Save
    output_path = os.path.join(output_dir, f"generation_{generation}_visualization.png")
    final_img.save(output_path)
    
    logger.info(f"Saved generation visualization to {output_path}")

def create_fitness_plot(
    best_fitness_history: list,
    avg_fitness_history: list,
    output_dir: str
):
    """
    Create a plot of fitness over generations.
    
    Args:
        best_fitness_history: List of best fitness scores per generation
        avg_fitness_history: List of average fitness scores per generation
        output_dir: Directory to save output
    """
    logger.info("Creating fitness history plot...")
    
    try:
        import matplotlib.pyplot as plt
        
        generations = list(range(len(best_fitness_history)))
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness_history, 'b-', label='Best Fitness')
        plt.plot(generations, avg_fitness_history, 'r--', label='Average Fitness')
        
        plt.title('Fitness Evolution Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.grid(True)
        plt.legend()
        
        output_path = os.path.join(output_dir, "fitness_history.png")
        plt.savefig(output_path)
        
        logger.info(f"Saved fitness history plot to {output_path}")
    except ImportError:
        logger.warning("Could not create fitness plot - matplotlib not available")
    except Exception as e:
        logger.error(f"Error creating fitness plot: {e}")

def main():
    """Run the multi-generation evolution example."""
    print("\n===== Anime Character Evolution - Multi-Generation Example =====\n")
    
    # Parameters
    generations = 5
    population_size = 8
    output_dir = "outputs/examples/multi_generation"
    
    # Run evolution
    run_multi_generation_evolution(
        generations=generations,
        population_size=population_size,
        output_dir=output_dir
    )
    
    print(f"\nEvolution complete! Results saved to {output_dir}")
    print("\nSummary:")
    print(f"- Generations: {generations}")
    print(f"- Population size: {population_size}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())