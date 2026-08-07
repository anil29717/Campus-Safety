import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import LiveCamera from "./pages/LiveCamera";
import Persons from "./pages/Persons";
import Objects from "./pages/Objects";
import Entries from "./pages/Entries";
import Activity from "./pages/Activity";
import Incidents from "./pages/Incidents";
import Safety from "./pages/Safety";
import Cameras from "./pages/Cameras";
import Features from "./pages/Features";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="camera" element={<LiveCamera />} />
        <Route path="cameras" element={<Cameras />} />
        <Route path="features" element={<Features />} />
        <Route path="persons" element={<Persons />} />
        <Route path="objects" element={<Objects />} />
        <Route path="entries" element={<Entries />} />
        <Route path="activity" element={<Activity />} />
        <Route path="incidents" element={<Incidents />} />
        <Route path="safety" element={<Safety />} />
      </Route>
    </Routes>
  );
}
