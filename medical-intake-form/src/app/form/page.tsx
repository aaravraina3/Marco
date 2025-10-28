"use client";

import { useState, useEffect } from "react";

interface MedicalFormData {
  patientName: string;
  dateOfBirth: string;
  phoneNumber: string;
  medicalConditions: string;
  currentMedications: string;
  allergies: string;
  emergencyContact: string;
  emergencyPhone: string;
}

export default function MedicalIntakeForm() {
  const [formData, setFormData] = useState<MedicalFormData>({
    patientName: "",
    dateOfBirth: "",
    phoneNumber: "",
    medicalConditions: "",
    currentMedications: "",
    allergies: "",
    emergencyContact: "",
    emergencyPhone: "",
  });

  const [hoveredField, setHoveredField] = useState<string | null>(null);
  const [activeField, setActiveField] = useState<string | null>(null);
  const [hoverTimeout, setHoverTimeout] = useState<NodeJS.Timeout | null>(null);

  const fields = [
    { id: "patientName", label: "Patient Name", type: "text", placeholder: "Enter your full name" },
    { id: "dateOfBirth", label: "Date of Birth", type: "text", placeholder: "e.g., January 15, 1985 or 01/15/1985" },
    { id: "phoneNumber", label: "Phone Number", type: "tel", placeholder: "Enter your phone number" },
    { id: "medicalConditions", label: "Medical Conditions", type: "textarea", placeholder: "List any existing medical conditions" },
    { id: "currentMedications", label: "Current Medications", type: "textarea", placeholder: "List all current medications and dosages" },
    { id: "allergies", label: "Allergies", type: "textarea", placeholder: "List any known allergies" },
    { id: "emergencyContact", label: "Emergency Contact Name", type: "text", placeholder: "Full name of emergency contact" },
    { id: "emergencyPhone", label: "Emergency Contact Phone", type: "tel", placeholder: "Emergency contact phone number" },
  ];

  useEffect(() => {
    return () => {
      if (hoverTimeout) {
        clearTimeout(hoverTimeout);
      }
    };
  }, [hoverTimeout]);

  const handleMouseEnter = (fieldId: string) => {
    setHoveredField(fieldId);
    const timeout = setTimeout(() => {
      setActiveField(fieldId);
      const element = document.getElementById(fieldId);
      if (element) {
        element.focus();
      }
    }, 1500); // 1.5 seconds hover to activate
    setHoverTimeout(timeout);
  };

  const handleMouseLeave = () => {
    setHoveredField(null);
    if (hoverTimeout) {
      clearTimeout(hoverTimeout);
      setHoverTimeout(null);
    }
  };

  const handleInputChange = (fieldId: string, value: string) => {
    setFormData((prev) => ({
      ...prev,
      [fieldId]: value,
    }));
  };

  const handleSkip = (fieldId: string) => {
    const currentIndex = fields.findIndex((f) => f.id === fieldId);
    if (currentIndex < fields.length - 1) {
      const nextField = fields[currentIndex + 1];
      setActiveField(nextField.id);
      const element = document.getElementById(nextField.id);
      if (element) {
        element.focus();
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Form submitted:", formData);
    alert("Medical intake form submitted successfully!");
  };

  const renderField = (field: { id: string; label: string; type: string; placeholder: string }) => {
    const isActive = activeField === field.id;
    const isHovered = hoveredField === field.id;
    const isFilled = formData[field.id as keyof MedicalFormData] !== "";

    return (
      <div
        key={field.id}
        className={`mb-8 p-6 rounded-lg border-4 transition-all duration-300 ${
          isActive
            ? "border-blue-500 bg-blue-50 shadow-lg"
            : isHovered
            ? "border-yellow-400 bg-yellow-50"
            : isFilled
            ? "border-green-500 bg-green-50"
            : "border-gray-300 bg-white"
        }`}
        onMouseEnter={() => handleMouseEnter(field.id)}
        onMouseLeave={handleMouseLeave}
      >
        <div className="flex items-center justify-between mb-4">
          <label htmlFor={field.id} className="text-2xl font-bold text-gray-800">
            {field.label}
            {isFilled && (
              <span className="ml-3 text-green-600 text-xl">✓ Filled</span>
            )}
            {isActive && (
              <span className="ml-3 text-blue-600 text-xl">● Active</span>
            )}
          </label>
        </div>

        {field.type === "textarea" ? (
          <textarea
            id={field.id}
            value={formData[field.id as keyof MedicalFormData]}
            onChange={(e) => handleInputChange(field.id, e.target.value)}
            placeholder={field.placeholder}
            className="w-full p-4 text-xl border-2 border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 resize-vertical min-h-[120px]"
            rows={4}
          />
        ) : (
          <input
            id={field.id}
            type={field.type}
            value={formData[field.id as keyof MedicalFormData]}
            onChange={(e) => handleInputChange(field.id, e.target.value)}
            placeholder={field.placeholder}
            className="w-full p-4 text-xl border-2 border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
          />
        )}

        {isActive && (
          <button
            type="button"
            onClick={() => handleSkip(field.id)}
            className="mt-4 px-6 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors text-lg"
          >
            Skip This Field
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
          <h1 className="text-5xl font-bold text-gray-900 mb-4 text-center">
            Medical Intake Form
          </h1>
          <p className="text-xl text-gray-600 text-center mb-8">
            Please hover over each field for 1.5 seconds to activate it. The field will turn blue when ready for input.
          </p>

          <form onSubmit={handleSubmit}>
            {fields.map((field) => renderField(field))}

            <div className="mt-12 text-center">
              <button
                type="submit"
                className="bg-blue-600 text-white text-2xl font-bold py-6 px-16 rounded-xl hover:bg-blue-700 transition-colors duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                Submit Medical Information
              </button>
            </div>
          </form>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Form Progress
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {fields.map((field) => (
              <div
                key={field.id}
                className={`p-3 rounded-lg text-center ${
                  formData[field.id as keyof MedicalFormData]
                    ? "bg-green-100 text-green-800"
                    : "bg-gray-100 text-gray-600"
                }`}
              >
                {formData[field.id as keyof MedicalFormData] ? "✓" : "○"} {field.label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

