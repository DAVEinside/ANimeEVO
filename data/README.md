# Data Directory

This directory contains various data files used by the Anime Character Evolution System.

## Subdirectories

- **character_templates/**: Contains predefined character templates that can be used as starting points for character creation
- **style_references/**: Contains reference images for various anime styles used in style transfer
- **user_creations/**: Contains user-created characters and configurations

## Adding Character Templates

To add a new character template, create a JSON file in the `character_templates` directory with the following format:

```json
{
  "name": "Template Name",
  "gender": "female",
  "age_category": "teen",
  "hair_color": "blue",
  "eye_color": "green",
  "personality": ["cheerful", "brave"],
  "distinctive_features": ["glasses"],
  "anime_style": "shojo",
  "art_style": "detailed",
  "era": "modern"
}
```

## Adding Style References

To add a new style reference, add an image file to the `style_references` directory and update the `styles.yaml` file in the config directory with the appropriate metadata.

## User Creations

The `user_creations` directory will be populated automatically when users save their characters through the interface.