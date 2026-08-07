import FeatureControlPanel from "../components/FeatureControlPanel";

export default function Features() {
  return (
    <div>
      <header className="mb-8">
        <h2 className="text-2xl font-bold text-white">Feature Control</h2>
        <p className="text-slate-400">
          Turn detection modules on or off and control the camera from one place.
        </p>
      </header>
      <FeatureControlPanel />
    </div>
  );
}
