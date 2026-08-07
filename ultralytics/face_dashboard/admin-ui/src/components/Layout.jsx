import {
  LayoutDashboard,
  Video,
  Users,
  Package,
  DoorOpen,
  Brain,
  AlertTriangle,
  Shield,
  Zap,
  Cctv,
} from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

const nav = [
  { to: "/", label: "Dashboard", Icon: LayoutDashboard },
  { to: "/camera", label: "Live Camera", Icon: Video },
  { to: "/cameras", label: "Camera Roles", Icon: Cctv },
  { to: "/features", label: "Features", Icon: Zap },
  { to: "/persons", label: "Persons", Icon: Users },
  { to: "/objects", label: "Objects", Icon: Package },
  { to: "/entries", label: "Entry History", Icon: DoorOpen },
  { to: "/activity", label: "Emotion & Behavior", Icon: Brain },
  { to: "/incidents", label: "Incidents", Icon: AlertTriangle },
  { to: "/safety", label: "Safety Zones", Icon: Shield },
];

export default function Layout() {
  return (
    <div className="flex min-h-screen">
      <aside className="fixed inset-y-0 left-0 z-20 flex w-64 flex-col border-r border-campus-700 bg-campus-900">
        <div className="border-b border-campus-700 px-5 py-6">
          <div className="flex items-center gap-2">
            <Shield className="h-6 w-6 text-campus-accent" strokeWidth={2} />
            <div>
              <h1 className="text-lg font-bold text-white">Campus Safety</h1>
              <p className="text-xs text-slate-500">Admin Console</p>
            </div>
          </div>
        </div>
        <nav className="flex-1 space-y-1 p-3">
          {nav.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition ${
                  isActive
                    ? "bg-campus-accent/20 text-campus-accent"
                    : "text-slate-400 hover:bg-campus-800 hover:text-white"
                }`
              }
            >
              <Icon className="h-[18px] w-[18px] shrink-0" strokeWidth={2} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-campus-700 p-4 text-xs text-slate-600">
          Sanskrit University · AI Monitor
        </div>
      </aside>
      <main className="ml-64 flex-1 p-8">
        <Outlet />
      </main>
    </div>
  );
}
