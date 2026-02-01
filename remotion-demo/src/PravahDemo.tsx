import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate, Sequence, spring, useVideoConfig } from 'remotion';

const colors = {
  primary: '#3B82F6',
  secondary: '#8B5CF6',
  background: '#0F172A',
  text: '#F1F5F9',
  accent: '#10B981',
  cardBg: '#1E293B',
};

// Title Scene
const TitleScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(frame, [0, 20], [0, 1]);
  const titleY = interpolate(frame, [0, 30], [50, 0]);
  const subtitleOpacity = interpolate(frame, [20, 40], [0, 1]);

  return (
    <AbsoluteFill style={{ backgroundColor: colors.background, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column' }}>
      <div style={{ opacity: titleOpacity, transform: `translateY(${titleY}px)`, textAlign: 'center' }}>
        <h1 style={{ fontSize: 96, margin: 0, background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%)`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontWeight: 'bold' }}>
          Pravah
        </h1>
        <div style={{ opacity: subtitleOpacity, marginTop: 20 }}>
          <p style={{ fontSize: 32, color: colors.text, margin: 0 }}>
            AI-Powered Search Engine
          </p>
        </div>
      </div>
    </AbsoluteFill>
  );
};

// Feature Card Component
const FeatureCard: React.FC<{ title: string; description: string; icon: string; delay: number }> = ({ title, description, icon, delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = spring({
    frame: frame - delay,
    fps,
    config: {
      damping: 100,
      stiffness: 200,
    },
  });

  const opacity = interpolate(frame - delay, [0, 10], [0, 1], { extrapolateLeft: 'clamp' });

  return (
    <div style={{
      transform: `scale(${scale})`,
      opacity,
      backgroundColor: colors.cardBg,
      borderRadius: 16,
      padding: 32,
      margin: 16,
      width: 300,
      border: `2px solid ${colors.primary}40`,
    }}>
      <div style={{ fontSize: 48, marginBottom: 16 }}>{icon}</div>
      <h3 style={{ fontSize: 28, color: colors.primary, margin: '0 0 12px 0' }}>{title}</h3>
      <p style={{ fontSize: 18, color: colors.text, margin: 0, lineHeight: 1.5 }}>{description}</p>
    </div>
  );
};

// Features Scene
const FeaturesScene: React.FC = () => {
  const features = [
    { title: 'Web Search', description: 'Tavily & Gemini search integration', icon: '🔍', delay: 0 },
    { title: 'Smart Fetching', description: 'Auto-summarize long pages', icon: '📄', delay: 10 },
    { title: 'Memory Search', description: 'Search fetched content', icon: '🧠', delay: 20 },
    { title: 'Multi-Model', description: 'GPT-4, Claude, Gemini & more', icon: '🤖', delay: 30 },
  ];

  return (
    <AbsoluteFill style={{ backgroundColor: colors.background, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40 }}>
      <div>
        <h2 style={{ fontSize: 48, color: colors.text, textAlign: 'center', marginBottom: 40 }}>Key Features</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
          {features.map((feature, i) => (
            <FeatureCard key={i} {...feature} />
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

// Agent Flow Visualization
const AgentFlowScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const steps = [
    { label: 'User Query', color: colors.primary, delay: 0 },
    { label: 'web_search', color: colors.accent, delay: 15 },
    { label: 'fetch_page', color: colors.accent, delay: 30 },
    { label: 'search_memory', color: colors.accent, delay: 45 },
    { label: 'AI Response', color: colors.secondary, delay: 60 },
  ];

  return (
    <AbsoluteFill style={{ backgroundColor: colors.background, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', padding: 60 }}>
      <h2 style={{ fontSize: 48, color: colors.text, marginBottom: 60 }}>Agent Workflow</h2>
      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        {steps.map((step, i) => {
          const opacity = interpolate(frame - step.delay, [0, 10], [0, 1], { extrapolateLeft: 'clamp' });
          const scale = spring({
            frame: frame - step.delay,
            fps,
            config: { damping: 100, stiffness: 200 },
          });

          return (
            <React.Fragment key={i}>
              <div style={{
                opacity,
                transform: `scale(${scale})`,
                backgroundColor: step.color,
                borderRadius: 12,
                padding: '20px 32px',
                color: 'white',
                fontSize: 20,
                fontWeight: 'bold',
                minWidth: 150,
                textAlign: 'center',
              }}>
                {step.label}
              </div>
              {i < steps.length - 1 && (
                <div style={{
                  opacity,
                  width: 30,
                  height: 3,
                  backgroundColor: colors.text,
                }}>
                  →
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

// Tools Showcase
const ToolsScene: React.FC = () => {
  const frame = useCurrentFrame();
  
  const tools = [
    { name: 'web_search', desc: 'Tavily web search', delay: 0 },
    { name: 'gemini_search', desc: 'Google Gemini grounded search', delay: 12 },
    { name: 'fetch_page', desc: 'Fetch & extract page content', delay: 24 },
    { name: 'read_page_chunk', desc: 'Navigate long documents', delay: 36 },
    { name: 'search_memory', desc: 'Search fetched content', delay: 48 },
    { name: 'calculate', desc: 'Math calculations', delay: 60 },
  ];

  return (
    <AbsoluteFill style={{ backgroundColor: colors.background, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 60 }}>
      <div style={{ maxWidth: 900 }}>
        <h2 style={{ fontSize: 48, color: colors.text, textAlign: 'center', marginBottom: 40 }}>Agent Tools</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
          {tools.map((tool, i) => {
            const opacity = interpolate(frame - tool.delay, [0, 10], [0, 1], { extrapolateLeft: 'clamp' });
            const x = interpolate(frame - tool.delay, [0, 15], [-50, 0], { extrapolateLeft: 'clamp' });

            return (
              <div key={i} style={{
                opacity,
                transform: `translateX(${x}px)`,
                backgroundColor: colors.cardBg,
                borderRadius: 12,
                padding: 24,
                borderLeft: `4px solid ${colors.accent}`,
              }}>
                <div style={{ fontSize: 22, color: colors.primary, fontWeight: 'bold', fontFamily: 'monospace', marginBottom: 8 }}>
                  {tool.name}
                </div>
                <div style={{ fontSize: 16, color: colors.text }}>
                  {tool.desc}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};

// End Scene with CTA
const EndScene: React.FC = () => {
  const frame = useCurrentFrame();
  
  const opacity = interpolate(frame, [0, 20], [0, 1]);

  return (
    <AbsoluteFill style={{ backgroundColor: colors.background, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column' }}>
      <div style={{ opacity, textAlign: 'center' }}>
        <h2 style={{ fontSize: 64, margin: 0, background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%)`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontWeight: 'bold' }}>
          Try Pravah
        </h2>
        <p style={{ fontSize: 28, color: colors.text, marginTop: 20 }}>
          uv run streamlit run app.py
        </p>
        <div style={{ fontSize: 20, color: colors.text, marginTop: 40, opacity: 0.7 }}>
          github.com/jayshah5696/pravah
        </div>
      </div>
    </AbsoluteFill>
  );
};

// Main Composition
export const PravahDemo: React.FC = () => {
  return (
    <AbsoluteFill>
      <Sequence from={0} durationInFrames={90}>
        <TitleScene />
      </Sequence>
      <Sequence from={90} durationInFrames={90}>
        <FeaturesScene />
      </Sequence>
      <Sequence from={180} durationInFrames={90}>
        <AgentFlowScene />
      </Sequence>
      <Sequence from={270} durationInFrames={90}>
        <ToolsScene />
      </Sequence>
      <Sequence from={360} durationInFrames={90}>
        <EndScene />
      </Sequence>
    </AbsoluteFill>
  );
};
