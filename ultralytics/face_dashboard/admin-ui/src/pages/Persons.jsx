import { useEffect, useState } from "react";
import { UserPlus, User, Trash2, UserX } from "lucide-react";
import { api } from "../api/client";
import AddFaceModal from "../components/AddFaceModal";

function PersonCard({ person, onDelete, unknown = false }) {
  return (
    <div className="card flex flex-col">
      {person.image_base64 ? (
        <img
          src={`data:image/jpeg;base64,${person.image_base64}`}
          alt={person.name || person.id}
          className="mb-3 h-32 w-full rounded-lg object-cover"
        />
      ) : (
        <div className="mb-3 flex h-32 items-center justify-center rounded-lg bg-campus-800">
          {unknown ? (
            <UserX className="h-12 w-12 text-amber-600/70" />
          ) : (
            <User className="h-12 w-12 text-slate-600" />
          )}
        </div>
      )}
      <h3 className="font-semibold text-white">
        {unknown ? "Unknown visitor" : person.name}
      </h3>
      <p className="mt-1 font-mono text-xs text-campus-accent">{person.id}</p>
      {unknown ? (
        <p className="mt-1 text-xs text-slate-500">
          Visits: {person.visit_count} · Last seen{" "}
          {person.last_seen ? new Date(person.last_seen).toLocaleString() : "—"}
        </p>
      ) : (
        <p className="truncate text-xs text-slate-500">
          {person.created_at ? new Date(person.created_at).toLocaleString() : ""}
        </p>
      )}
      <button
        type="button"
        className="btn-ghost mt-3 flex items-center justify-center gap-2 text-campus-danger"
        onClick={() => onDelete(person.id)}
      >
        <Trash2 className="h-4 w-4" />
        Delete
      </button>
    </div>
  );
}

export default function Persons() {
  const [persons, setPersons] = useState([]);
  const [unknowns, setUnknowns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const [known, unknown] = await Promise.all([
        api.persons(),
        api.unknownPersons(),
      ]);
      setPersons(known);
      setUnknowns(unknown);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const removeKnown = async (id) => {
    if (!confirm("Delete this registered person?")) return;
    await api.deletePerson(id);
    load();
  };

  const removeUnknown = async (id) => {
    if (!confirm("Delete this unknown visitor record and photo?")) return;
    await api.deleteUnknownPerson(id);
    load();
  };

  return (
    <div>
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Persons</h2>
          <p className="text-slate-400">
            Registered faces and unknown visitors captured by the camera
          </p>
        </div>
        <button
          type="button"
          className="btn-primary flex items-center gap-2"
          onClick={() => setModalOpen(true)}
        >
          <UserPlus className="h-4 w-4" />
          Add face
        </button>
      </header>

      <AddFaceModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onSuccess={load}
      />

      {loading ? (
        <p className="text-slate-500">Loading…</p>
      ) : (
        <>
          <section className="mb-10">
            <h3 className="mb-4 text-lg font-semibold text-white">
              Registered persons ({persons.length})
            </h3>
            {persons.length === 0 ? (
              <p className="text-slate-500">
                No registered persons yet. Click <strong>Add face</strong> to register
                someone.
              </p>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {persons.map((p) => (
                  <PersonCard key={p.id} person={p} onDelete={removeKnown} />
                ))}
              </div>
            )}
          </section>

          <section>
            <h3 className="mb-1 text-lg font-semibold text-white">
              Unknown visitors ({unknowns.length})
            </h3>
            <p className="mb-4 text-sm text-slate-500">
              When an unrecognized face appears, the system assigns an ID and saves a
              photo automatically.
            </p>
            {unknowns.length === 0 ? (
              <p className="text-slate-500">
                No unknown visitors recorded yet. Start the live camera to detect them.
              </p>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {unknowns.map((p) => (
                  <PersonCard
                    key={p.id}
                    person={p}
                    onDelete={removeUnknown}
                    unknown
                  />
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
