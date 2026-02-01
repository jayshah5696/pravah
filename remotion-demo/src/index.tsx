import { registerRoot } from 'remotion';
import { Composition } from 'remotion';
import { PravahDemo } from './PravahDemo';

export const RemotionRoot = () => {
  return (
    <>
      <Composition
        id="PravahDemo"
        component={PravahDemo}
        durationInFrames={450} // 15 seconds at 30fps
        fps={30}
        width={1200}
        height={675}
        defaultProps={{}}
      />
    </>
  );
};

registerRoot(RemotionRoot);
