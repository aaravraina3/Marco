import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-gray-900 mb-8">
          Healthcare Intake Assistant
        </h1>
        <p className="text-2xl text-gray-600 mb-12 max-w-2xl mx-auto">
          Accessible medical intake form with voice navigation
        </p>
        <Link
          href="/form"
          className="inline-block bg-blue-600 text-white text-2xl font-bold py-6 px-12 rounded-xl hover:bg-blue-700 transition-colors duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
        >
          Start Medical Intake Form
        </Link>
      </div>
    </div>
  );
}

