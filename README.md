# Healthcare-Intake-Assistant

Open CV multi-modal healthcare intake system that adapts to any patient's abilities

## Medical Intake Form (Next.js)

This is a [Next.js](https://nextjs.org) medical intake form with eye tracking accessibility features.

### Features

- **Complete Medical Form**: Patient info, medical conditions, medications, allergies, emergency contacts
- **Eye Tracking Accessibility**: 1.5-second hover detection with blue glow animation
- **Large Targets**: Extra large input fields (60px+ height) and huge click targets
- **High Contrast**: Professional medical styling with accessible colors
- **Responsive Design**: Works on all devices with Tailwind CSS

### Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

### Eye Tracking Integration

The form is designed to work with eye tracking systems:
- Hover over any field for 1.5 seconds to activate it
- Fields glow blue when ready for eye tracking interaction
- Large targets make it easy for eye tracking to hit accurately
- Click events can be triggered by blink detection

### Python Eye Tracking System

The companion Python system provides:
- Real-time eye tracking with MediaPipe
- Facial paralysis detection using FaCiPa algorithm
- Tremor amplitude measurement
- Distance-based camera zoom control
- Form navigation integration

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.