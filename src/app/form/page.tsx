'use client'

import { useState, useRef, useEffect } from 'react'

export default function MedicalIntakeForm() {
  const [formData, setFormData] = useState({
    patientName: '',
    dateOfBirth: '',
    phoneNumber: '',
    medicalConditions: '',
    currentMedications: '',
    allergies: '',
    emergencyContact: '',
    emergencyPhone: ''
  })

  const [gazeReadyFields, setGazeReadyFields] = useState<Set<string>>(new Set())
  const hoverTimers = useRef<Map<string, NodeJS.Timeout>>(new Map())

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const handleMouseEnter = (fieldName: string) => {
    // Clear any existing timer for this field
    const existingTimer = hoverTimers.current.get(fieldName)
    if (existingTimer) {
      clearTimeout(existingTimer)
    }

    // Start new timer
    const timer = setTimeout(() => {
      setGazeReadyFields(prev => new Set(prev).add(fieldName))
    }, 1500) // 1.5 seconds

    hoverTimers.current.set(fieldName, timer)
  }

  const handleMouseLeave = (fieldName: string) => {
    // Clear timer
    const timer = hoverTimers.current.get(fieldName)
    if (timer) {
      clearTimeout(timer)
      hoverTimers.current.delete(fieldName)
    }

    // Remove gaze-ready state
    setGazeReadyFields(prev => {
      const newSet = new Set(prev)
      newSet.delete(fieldName)
      return newSet
    })
  }

  const handleFieldClick = (fieldName: string) => {
    // Focus the field and remove gaze-ready state
    const element = document.getElementById(fieldName)
    if (element) {
      element.focus()
    }
    setGazeReadyFields(prev => {
      const newSet = new Set(prev)
      newSet.delete(fieldName)
      return newSet
    })
  }

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      hoverTimers.current.forEach(timer => clearTimeout(timer))
    }
  }, [])

  const inputClasses = `
    w-full p-4 text-lg border-2 border-gray-300 rounded-lg
    focus:border-blue-500 focus:ring-2 focus:ring-blue-200 focus:outline-none
    transition-all duration-200 min-h-[60px] bg-white
    text-gray-900 placeholder-gray-500
  `

  const textareaClasses = `
    w-full p-4 text-lg border-2 border-gray-300 rounded-lg
    focus:border-blue-500 focus:ring-2 focus:ring-blue-200 focus:outline-none
    transition-all duration-200 min-h-[100px] bg-white
    text-gray-900 placeholder-gray-500 resize-vertical
  `

  const gazeReadyClasses = `
    animate-pulse shadow-lg shadow-blue-500/50
    border-blue-400 bg-blue-50
  `

  const fieldContainerClasses = (fieldName: string) => `
    mb-8 p-6 rounded-xl transition-all duration-300
    ${gazeReadyFields.has(fieldName) 
      ? 'bg-blue-100 border-4 border-blue-400 shadow-lg' 
      : 'bg-gray-50 border-2 border-gray-200 hover:bg-gray-100'
    }
  `

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Medical Intake Form
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Please fill out this form completely. Hover over any field for 1.5 seconds to activate it for eye tracking.
          </p>
        </div>

        <form className="bg-white rounded-2xl shadow-2xl p-8">
          {/* Patient Name */}
          <div 
            className={fieldContainerClasses('patientName')}
            onMouseEnter={() => handleMouseEnter('patientName')}
            onMouseLeave={() => handleMouseLeave('patientName')}
            onClick={() => handleFieldClick('patientName')}
          >
            <label htmlFor="patientName" className="block text-xl font-semibold text-gray-900 mb-3">
              Patient Name *
            </label>
            <input
              id="patientName"
              type="text"
              className={`${inputClasses} ${gazeReadyFields.has('patientName') ? gazeReadyClasses : ''}`}
              placeholder="Enter full name"
              value={formData.patientName}
              onChange={(e) => handleInputChange('patientName', e.target.value)}
              required
            />
          </div>

          {/* Date of Birth */}
          <div 
            className={fieldContainerClasses('dateOfBirth')}
            onMouseEnter={() => handleMouseEnter('dateOfBirth')}
            onMouseLeave={() => handleMouseLeave('dateOfBirth')}
            onClick={() => handleFieldClick('dateOfBirth')}
          >
            <label htmlFor="dateOfBirth" className="block text-xl font-semibold text-gray-900 mb-3">
              Date of Birth *
            </label>
            <input
              id="dateOfBirth"
              type="date"
              className={`${inputClasses} ${gazeReadyFields.has('dateOfBirth') ? gazeReadyClasses : ''}`}
              value={formData.dateOfBirth}
              onChange={(e) => handleInputChange('dateOfBirth', e.target.value)}
              required
            />
          </div>

          {/* Phone Number */}
          <div 
            className={fieldContainerClasses('phoneNumber')}
            onMouseEnter={() => handleMouseEnter('phoneNumber')}
            onMouseLeave={() => handleMouseLeave('phoneNumber')}
            onClick={() => handleFieldClick('phoneNumber')}
          >
            <label htmlFor="phoneNumber" className="block text-xl font-semibold text-gray-900 mb-3">
              Phone Number *
            </label>
            <input
              id="phoneNumber"
              type="tel"
              className={`${inputClasses} ${gazeReadyFields.has('phoneNumber') ? gazeReadyClasses : ''}`}
              placeholder="(555) 123-4567"
              value={formData.phoneNumber}
              onChange={(e) => handleInputChange('phoneNumber', e.target.value)}
              required
            />
          </div>

          {/* Medical Conditions */}
          <div 
            className={fieldContainerClasses('medicalConditions')}
            onMouseEnter={() => handleMouseEnter('medicalConditions')}
            onMouseLeave={() => handleMouseLeave('medicalConditions')}
            onClick={() => handleFieldClick('medicalConditions')}
          >
            <label htmlFor="medicalConditions" className="block text-xl font-semibold text-gray-900 mb-3">
              Medical Conditions
            </label>
            <textarea
              id="medicalConditions"
              className={`${textareaClasses} ${gazeReadyFields.has('medicalConditions') ? gazeReadyClasses : ''}`}
              placeholder="List any current medical conditions, diagnoses, or health concerns..."
              value={formData.medicalConditions}
              onChange={(e) => handleInputChange('medicalConditions', e.target.value)}
              rows={4}
            />
          </div>

          {/* Current Medications */}
          <div 
            className={fieldContainerClasses('currentMedications')}
            onMouseEnter={() => handleMouseEnter('currentMedications')}
            onMouseLeave={() => handleMouseLeave('currentMedications')}
            onClick={() => handleFieldClick('currentMedications')}
          >
            <label htmlFor="currentMedications" className="block text-xl font-semibold text-gray-900 mb-3">
              Current Medications
            </label>
            <textarea
              id="currentMedications"
              className={`${textareaClasses} ${gazeReadyFields.has('currentMedications') ? gazeReadyClasses : ''}`}
              placeholder="List all current medications, including dosage and frequency..."
              value={formData.currentMedications}
              onChange={(e) => handleInputChange('currentMedications', e.target.value)}
              rows={4}
            />
          </div>

          {/* Allergies */}
          <div 
            className={fieldContainerClasses('allergies')}
            onMouseEnter={() => handleMouseEnter('allergies')}
            onMouseLeave={() => handleMouseLeave('allergies')}
            onClick={() => handleFieldClick('allergies')}
          >
            <label htmlFor="allergies" className="block text-xl font-semibold text-gray-900 mb-3">
              Allergies
            </label>
            <textarea
              id="allergies"
              className={`${textareaClasses} ${gazeReadyFields.has('allergies') ? gazeReadyClasses : ''}`}
              placeholder="List any allergies to medications, foods, or other substances..."
              value={formData.allergies}
              onChange={(e) => handleInputChange('allergies', e.target.value)}
              rows={4}
            />
          </div>

          {/* Emergency Contact */}
          <div 
            className={fieldContainerClasses('emergencyContact')}
            onMouseEnter={() => handleMouseEnter('emergencyContact')}
            onMouseLeave={() => handleMouseLeave('emergencyContact')}
            onClick={() => handleFieldClick('emergencyContact')}
          >
            <label htmlFor="emergencyContact" className="block text-xl font-semibold text-gray-900 mb-3">
              Emergency Contact Name *
            </label>
            <input
              id="emergencyContact"
              type="text"
              className={`${inputClasses} ${gazeReadyFields.has('emergencyContact') ? gazeReadyClasses : ''}`}
              placeholder="Full name of emergency contact"
              value={formData.emergencyContact}
              onChange={(e) => handleInputChange('emergencyContact', e.target.value)}
              required
            />
          </div>

          {/* Emergency Phone */}
          <div 
            className={fieldContainerClasses('emergencyPhone')}
            onMouseEnter={() => handleMouseEnter('emergencyPhone')}
            onMouseLeave={() => handleMouseLeave('emergencyPhone')}
            onClick={() => handleFieldClick('emergencyPhone')}
          >
            <label htmlFor="emergencyPhone" className="block text-xl font-semibold text-gray-900 mb-3">
              Emergency Contact Phone *
            </label>
            <input
              id="emergencyPhone"
              type="tel"
              className={`${inputClasses} ${gazeReadyFields.has('emergencyPhone') ? gazeReadyClasses : ''}`}
              placeholder="(555) 123-4567"
              value={formData.emergencyPhone}
              onChange={(e) => handleInputChange('emergencyPhone', e.target.value)}
              required
            />
          </div>

          {/* Submit Button */}
          <div className="mt-12 text-center">
            <button
              type="submit"
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold text-xl py-4 px-12 rounded-xl
                         transition-all duration-200 min-w-[200px] min-h-[60px]
                         shadow-lg hover:shadow-xl focus:outline-none focus:ring-4 focus:ring-blue-300"
            >
              Submit Form
            </button>
          </div>
        </form>

        {/* Instructions */}
        <div className="mt-8 text-center text-gray-600">
          <p className="text-lg">
            💡 <strong>Eye Tracking Tip:</strong> Hover over any field for 1.5 seconds to activate it. 
            The field will glow blue when ready for eye tracking interaction.
          </p>
        </div>
      </div>
    </div>
  )
}
