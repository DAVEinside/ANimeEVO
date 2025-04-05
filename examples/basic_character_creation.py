"""
Basic example of character creation and evolution.
This script demonstrates the core functionality of the anime character evolution system.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from core.attributes.character_attributes import CharacterAttributes
from core.diffusion.anime_pipeline import AnimePipeline
from core.evolution.evolution_engine import EvolutionEngine
from multimodal.image_generator import ImageGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the basic character creation and evolution example."""
    print("=== Anime Character Evolution System - Basic Example ===")
    
    # Create output directory
    output_dir = os.path.join("outputs", "examples")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Initialize core components
    print("\nInitializing system components...")
    pipeline = AnimePipeline()
    evolution_engine = EvolutionEngine()
    image_generator = ImageGenerator(pipeline=pipeline)
    
    # Step 2: Create a character
    print("\nCreating character...")
    
    # Create a character with specific attributes
    shojo_character = CharacterAttributes(
        name="Sakura",
        gender="female",
        age_category="teen",
        hair_color="pink",
        eye_color="green",
        personality=["cheerful", "determined", "kind"],
        anime_style="shojo",
        distinctive_features=["hair ribbon", "school uniform"]
    )
    
    # Generate character image
    print("Generating character image...")
    images, output = image_generator.generate_images(
        attributes=shojo_character,
        num_samples=2
    )
    
    # Save images
    image_paths = image_generator.save_images(
        images=images,
        character=shojo_character,
        output=output,
        prefix="example_original"
    )
    
    # Display results
    print(f"Character created: {shojo_character.name}")
    print(f"ID: {shojo_character.id}")
    print(f"Generated {len(image_paths)} images")
    
    # Step 3: Create a second character with different style
    print("\nCreating second character...")
    
    # Create a character with contrasting attributes
    shonen_character = CharacterAttributes(
        name="Ryu",
        gender="male",
        age_category="teen",
        hair_color="blue",
        eye_color="gold",
        personality=["serious", "brave", "reserved"],
        anime_style="shonen",
        distinctive_features=["scar", "headband"]
    )
    
    # Generate character image
    print("Generating character image...")
    images2, output2 = image_generator.generate_images(
        attributes=shonen_character,
        num_samples=2
    )
    
    # Save images
    image_paths2 = image_generator.save_images(
        images=images2,
        character=shonen_character,
        output=output2,
        prefix="example_original"
    )
    
    # Display results
    print(f"Character created: {shonen_character.name}")
    print(f"ID: {shonen_character.id}")
    print(f"Generated {len(image_paths2)} images")
    
    # Step 4: Evolve first character (mutation)
    print("\nEvolving first character (mutation)...")
    
    # Single evolution
    evolved_character1 = evolution_engine.evolve_single(shojo_character)
    
    # Generate evolved character image
    print("Generating evolved character image...")
    images_evolved1, output_evolved1 = image_generator.generate_images(
        attributes=evolved_character1,
        num_samples=1
    )
    
    # Save images
    image_paths_evolved1 = image_generator.save_images(
        images=images_evolved1,
        character=evolved_character1,
        output=output_evolved1,
        prefix="example_evolved_mutation"
    )
    
    # Display results
    print(f"Evolved character created (mutation)")
    print(f"ID: {evolved_character1.id}")
    print(f"Generation: {evolved_character1.generation}")
    print(f"Parent ID: {evolved_character1.parent_ids[0]}")
    
    # Step 5: Evolve by combining characters (crossover)
    print("\nEvolving by combining characters (crossover)...")
    
    # Combine characters
    evolved_character2 = evolution_engine.evolve_pair(shojo_character, shonen_character)
    
    # Generate evolved character image
    print("Generating evolved character image...")
    images_evolved2, output_evolved2 = image_generator.generate_images(
        attributes=evolved_character2,
        num_samples=1
    )
    
    # Save images
    image_paths_evolved2 = image_generator.save_images(
        images=images_evolved2,
        character=evolved_character2,
        output=output_evolved2,
        prefix="example_evolved_crossover"
    )
    
    # Display results
    print(f"Evolved character created (crossover)")
    print(f"ID: {evolved_character2.id}")
    print(f"Generation: {evolved_character2.generation}")
    print(f"Parent IDs: {', '.join(evolved_character2.parent_ids)}")
    
    # Step 6: Multiple generations of evolution
    print("\nPerforming multi-generation evolution...")
    
    # Create population
    population = [shojo_character]
    for _ in range(3):  # Add some variations
        population.append(shojo_character.mutate(0.3))
        
    # Evolve for multiple generations
    evolved_population = evolution_engine.evolve(
        population,
        generation_count=3
    )
    
    # Generate and save images for final population
    print(f"Generating images for {len(evolved_population)} evolved characters...")
    
    for i, character in enumerate(evolved_population):
        print(f"Character {i+1}, generation {character.generation}...")
        
        # Generate image
        images_gen, output_gen = image_generator.generate_images(
            attributes=character,
            num_samples=1
        )
        
        # Save image
        image_paths_gen = image_generator.save_images(
            images=images_gen,
            character=character,
            output=output_gen,
            prefix=f"example_gen{character.generation}_char{i}"
        )
    
    # Step 7: Generate character sheet for best character
    print("\nGenerating character sheet...")
    
    # Use the first evolution result
    character_sheet_path = image_generator.create_character_sheet(
        character=evolved_character2,
        images=images_evolved2,
        include_info=True,
        output_path=os.path.join(output_dir, "example_character_sheet.png")
    )
    
    print(f"Character sheet created: {character_sheet_path}")
    
    # Summary
    print("\n=== Example Complete ===")
    print("All output files have been saved to:", output_dir)
    print("\nCharacters created:")
    print(f"- Original: {shojo_character.name} (ID: {shojo_character.id[:8]})")
    print(f"- Original: {shonen_character.name} (ID: {shonen_character.id[:8]})")
    print(f"- Mutation: Evolved from {shojo_character.name} (ID: {evolved_character1.id[:8]})")
    print(f"- Crossover: Combined from both parents (ID: {evolved_character2.id[:8]})")
    print(f"- Population: {len(evolved_population)} characters evolved over 3 generations")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())