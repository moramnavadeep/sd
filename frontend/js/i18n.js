// UI strings. Spoken/guidance text comes from the server already translated (config/i18n/*.yaml).
// Telugu strings should be reviewed by native speakers during the pilot.
export const UI = {
  en: {
    back: '◀ Back', repeat: 'Repeat', confirm: 'Confirm step', check: 'Am I right?', next: 'Next ▶', voice: '🎤 Voice',
    st_ok: 'In progress', st_correction: 'Adjust', st_uncertain: 'Not sure — check yourself', st_need_position: 'Reposition', st_step_complete: 'Looks done',
    starting: 'Starting…', allow_camera: 'Allow camera access to begin.',
    camera_off: 'Camera unavailable ({err}). Voice commands and buttons still work — allow camera access and reload to see the mirror.',
    disconnected: 'Local server disconnected — retrying…',
    heard: 'Heard', not_command: 'not a command', not_connected: 'not connected to the local server yet',
    meter_paused: 'paused while I speak', meter_hearing: 'hearing you', meter_hint: 'say “next step”',
    language: 'Language',
    hosted_chip: 'SERVER PROCESSING',
    voice_on: 'ON', voice_off: 'OFF',
    mic_click: 'Click anywhere on the page once to switch the microphone on (browser rule).',
    mic_blocked: 'Microphone blocked ({err}). Click the camera/mic icon in the address bar, allow the microphone, then reload.',
  },
  te: {
    back: '◀ వెనక్కి', repeat: 'మళ్లీ చెప్పు', confirm: 'దశ పూర్తి', check: 'సరిగ్గా ఉందా?', next: 'తరువాత ▶', voice: '🎤 వాయిస్',
    st_ok: 'కొనసాగుతోంది', st_correction: 'సరిచేయండి', st_uncertain: 'ఖచ్చితంగా తెలియదు — మీరే చూసుకోండి', st_need_position: 'స్థానం మార్చండి', st_step_complete: 'పూర్తైనట్లుంది',
    starting: 'ప్రారంభిస్తోంది…', allow_camera: 'ప్రారంభించడానికి కెమెరాకు అనుమతి ఇవ్వండి.',
    camera_off: 'కెమెరా అందుబాటులో లేదు ({err}). వాయిస్ ఆదేశాలు, బటన్లు పని చేస్తాయి — అద్దం చూడటానికి కెమెరాకు అనుమతి ఇచ్చి రీలోడ్ చేయండి.',
    disconnected: 'లోకల్ సర్వర్‌తో కనెక్షన్ పోయింది — మళ్లీ ప్రయత్నిస్తోంది…',
    heard: 'విన్నది', not_command: 'ఆదేశం కాదు', not_connected: 'లోకల్ సర్వర్‌కు ఇంకా కనెక్ట్ కాలేదు',
    meter_paused: 'నేను మాట్లాడుతున్నప్పుడు ఆగింది', meter_hearing: 'వింటున్నాను', meter_hint: '“తరువాత” అని చెప్పండి',
    language: 'భాష',
    hosted_chip: 'సర్వర్‌లో ప్రాసెసింగ్',
    voice_on: 'ఆన్', voice_off: 'ఆఫ్',
    mic_click: 'మైక్రోఫోన్ ఆన్ చేయడానికి పేజీలో ఎక్కడైనా ఒకసారి క్లిక్ చేయండి (బ్రౌజర్ నియమం).',
    mic_blocked: 'మైక్రోఫోన్ బ్లాక్ అయింది ({err}). అడ్రస్ బార్‌లోని కెమెరా/మైక్ గుర్తుపై క్లిక్ చేసి, మైక్రోఫోన్‌ను అనుమతించి, రీలోడ్ చేయండి.',
  },
};

export function tr(lang, key, vars = {}) {
  const s = (UI[lang] && UI[lang][key]) || UI.en[key] || key;
  return s.replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? '');
}
