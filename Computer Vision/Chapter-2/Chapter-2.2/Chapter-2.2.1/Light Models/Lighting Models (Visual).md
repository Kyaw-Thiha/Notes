# Lighting Models — Visual Overview
#cv/light #graphics 

This page provides **visual intuition diagrams** for the major lighting models in CV/graphics.  
Each diagram is schematic (not to scale) and links back to the detailed page.  

---

## Point Light
```
          * (Point source)
         /|\ 
        / | \
       /  |  \
      v   v   v
   -------------
    Surface plane
```
- Emits equally in all directions.  
- 🔗 See: [[Point Light Source]]  

---

## Area Light
```
     ┌───────────┐   (Rectangular area light)
     │           │
     │           │
     └───────────┘
        \   |   /
         \  |  /
          v v v
   -------------
    Surface plane
```
- Extended surface emitter → **soft shadows**.  
- 🔗 See: [[Area Light]]  

---

## Directional Light
```
    ↓   ↓   ↓   ↓   ↓
    ↓   ↓   ↓   ↓   ↓   (Parallel rays)
   -------------------
        Surface
```
- All rays parallel, constant intensity (sunlight).  
- 🔗 See: [[Directional Light]]  

---

## Spotlight
```
        * (Source position)
         \ 
          \  (cutoff angle θc)
           \ v
            \|
             v
        ------------
         Surface area
```
- Point source with restricted cone of emission.  
- 🔗 See: [[Spotlight]]  

---

## Environment Map
```
        +-------------------+
       /                     \
      /   o (Scene object)    \
     |                         |
     |                         |
      \                       /
       \                     /
        +-------------------+

Illumination comes from all directions of a surrounding sphere.
```
- Captures **directional radiance distribution**.  
- 🔗 See: [[Environment Map]]  

---

## HDR Environment Map
```
 Lat-Long HDR Panorama:
 +--------------------------------+
 |    ☼ (bright sun, high value)  |
 |                                |
 |   blue sky (medium value)      |
 |                                |
 |   dark ground (low value)      |
 +--------------------------------+
```
- Stores **high dynamic range values** (sun vs sky vs ground).  
- 🔗 See: [[HDR Environment Map]]  

---

## Sky Models
```
         ☼  (Sun at elevation θs)
       .-' `-.
     .'       `.
    /           \
   |   Sky dome  |
    \           /
     `.       .’
       `-._.-’

Radiance distribution depends on sun position + turbidity.
```
- Analytical daylight distribution over sky dome.  
- 🔗 See: [[Sky Models]]  

---

## Ambient Light
```
  ↘ ↓ ↙   ↘ ↓ ↙   ↘ ↓ ↙
   ↘ ↓ ↙   ↘ ↓ ↙   ↘ ↓ ↙
   -----------------------
        Surface plane
```
- Uniform light from all directions, constant everywhere.  
- 🔗 See: [[Ambient Light]]  

---

# Key Glance
- **Local emitters**: [[Point Light Source]], [[Area Light]], [[Spotlight]].  
- **Distant/global emitters**: [[Directional Light]], [[Environment Map]], [[HDR Environment Map]], [[Sky Models]].  
- **Simplified approximation**: [[Ambient Light]].  
