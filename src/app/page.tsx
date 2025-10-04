import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-5xl font-bold text-gray-900 mb-8">
          Healthcare Intake Assistant
        </h1>
        <p className="text-xl text-gray-600 mb-12 max-w-2xl mx-auto">
          Accessible medical intake forms designed for eye tracking technology. 
          Perfect for patients with mobility challenges or those who prefer hands-free interaction.
        </p>
        <div className="space-y-4">
          <Link
            href="/form"
            className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-bold text-xl py-4 px-12 rounded-xl
                       transition-all duration-200 shadow-lg hover:shadow-xl focus:outline-none focus:ring-4 focus:ring-blue-300"
          >
            Start Medical Intake Form
          </Link>
          <div className="text-sm text-gray-500">
            <p>💡 Eye tracking compatible • 🎯 Large targets • ♿ Accessibility focused</p>
          </div>
        </div>
      </div>
    </div>
  );
}