# Pravah Demo Video (Remotion)

This directory contains the Remotion project for generating the Pravah demo GIF.

## What's Included

The demo showcases:
- **Title Scene**: Pravah branding with gradient animation
- **Features Scene**: 4 key features with spring animations
  - Web Search (Tavily & Gemini)
  - Smart Fetching (auto-summarization)
  - Memory Search
  - Multi-Model Support
- **Agent Flow Scene**: Visual workflow of agent execution
- **Tools Scene**: All 6 agent tools with descriptions
- **End Scene**: Call-to-action with repository link

## Setup

```bash
npm install
```

## Preview

Preview the video composition in your browser:

```bash
npm start
```

This opens the Remotion Studio at http://localhost:3000

## Render

### Render to GIF (Optimized)

```bash
npm run gif -- --height=250 --width=440 --every-nth-frame=3
```

This creates `out/demo.gif` with:
- Dimensions: 440x250px
- Frame rate: ~10fps (every 3rd frame)
- File size: ~1.8MB
- Duration: 15 seconds

### Render to MP4

```bash
npm run build
```

This creates `out/demo.mp4` at full quality.

### Custom Render Options

```bash
# Higher quality GIF (larger file)
npm run gif -- --height=400 --width=700 --every-nth-frame=2

# Different format
remotion render src/index.tsx PravahDemo out/demo.webm --codec vp8

# Specific frame range
remotion render src/index.tsx PravahDemo out/demo.mp4 --frames=0-90
```

## Copy to Assets

After rendering, copy to the main assets folder:

```bash
cp out/demo.gif ../assets/demo.gif
```

## File Structure

```
remotion-demo/
├── src/
│   ├── index.tsx          # Remotion root and composition
│   └── PravahDemo.tsx     # Main demo component with scenes
├── out/                   # Rendered output (gitignored)
├── package.json
├── tsconfig.json
└── README.md
```

## Customization

To modify the demo:

1. Edit `src/PravahDemo.tsx` to change scenes, animations, or content
2. Adjust timing by modifying `durationInFrames` in scene `<Sequence>` components
3. Change colors in the `colors` object at the top of `PravahDemo.tsx`
4. Preview changes with `npm start`
5. Re-render with `npm run gif`

## Animation Techniques Used

- **Spring animations**: Natural, physics-based motion for cards and elements
- **Interpolation**: Smooth opacity and position transitions
- **Sequencing**: Scene transitions with proper timing
- **Gradient text**: Eye-catching title effects
- **Responsive layout**: Grid-based feature cards

## Learn More

- [Remotion Documentation](https://www.remotion.dev/docs)
- [Remotion Animation Guide](https://www.remotion.dev/docs/animating-properties)
- [Remotion Spring Animations](https://www.remotion.dev/docs/spring)
