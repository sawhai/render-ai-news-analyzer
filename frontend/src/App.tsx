import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, Download, CheckCircle, AlertCircle, Clock, FileText, Building2, Newspaper, BarChart3, Settings, X, Zap, Calendar, AlertTriangle, Archive, RefreshCw, Eye, Timer } from 'lucide-react';
import './App.css';

interface Bank {
  id: string;
  name: string;
  arabicName: string;
}

interface NewspaperScript {
  id: string;
  name: string;
  arabicName: string;
}

interface AnalysisResult {
  bank: string;
  newspaper: string;
  hasContent: boolean;
  error: boolean;
  pages: number;
  highlights: string[];
  errorMessage?: string;
}

interface Progress {
  current: number;
  total: number;
  currentTask: string;
  phase: 'idle' | 'downloading' | 'analyzing' | 'generating' | 'complete' | 'error';
}

interface Report {
  type: string;
  name: string;
  filename: string;
  download_url: string;
}

const App: React.FC = () => {
  // Bank and newspaper configurations
  const availableBanks: Bank[] = [
    { id: 'gulf_bank', name: 'Gulf Bank', arabicName: 'بنك الخليج' },
    { id: 'nbk', name: 'National Bank of Kuwait', arabicName: 'البنك الوطني الكويتي' },
    { id: 'kfh', name: 'Kuwait Finance House', arabicName: 'بيت التمويل الكويتي' },
    { id: 'cbk', name: 'Commercial Bank of Kuwait', arabicName: 'البنك التجاري الكويتي' },
    { id: 'burgan_bank', name: 'Burgan Bank', arabicName: 'بنك برقان' },
    { id: 'kib', name: 'Kuwait International Bank', arabicName: 'بنك الكويت الدولي' },
    { id: 'abk', name: 'Al Ahli Bank of Kuwait', arabicName: 'البنك الأهلي الكويتي' },
    { id: 'warba_bank', name: 'Warba Bank', arabicName: 'بنك وربة' },
    { id: 'boubyan_bank', name: 'Boubyan Bank', arabicName: 'بنك بوبيان' }
  ];

  const availableNewspapers: NewspaperScript[] = [
    { id: 'alrai_multibank', name: 'Al-Rai Media', arabicName: 'الراي' },
    { id: 'aljarida_multibank', name: 'Al-Jarida', arabicName: 'الجريدة' },
    { id: 'alqabas_multibank', name: 'Al-Qabas', arabicName: 'القبس' },
    { id: 'alnahar_multibank', name: 'Al-Nahar', arabicName: 'النهار' },
    { id: 'kwttimes_multibank', name: 'Kuwait Times', arabicName: 'كويت تايمز' },
    { id: 'arabtimes_multibank', name: 'Arab Times', arabicName: 'عرب تايمز' },
    { id: 'alwasat_multibank', name: 'Al-Wasat', arabicName: 'الوسط' },
    { id: 'alanbaa_multibanks', name: 'Al-Anbaa', arabicName: 'الأنباء' },
    { id: 'alseyassah_multibank', name: 'Al-Seyassah', arabicName: 'السياسة' }
  ];

  // Basic state
  const [selectedBanks, setSelectedBanks] = useState<Set<string>>(new Set(availableBanks.map(b => b.id)));
  const [selectedNewspapers, setSelectedNewspapers] = useState<Set<string>>(new Set(availableNewspapers.map(n => n.id)));
  const [selectedModel, setSelectedModel] = useState<'gpt-4o-mini' | 'gpt-4o'>('gpt-4o-mini');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [progress, setProgress] = useState<Progress>({
    current: 0,
    total: 0,
    currentTask: '',
    phase: 'idle'
  });
  const [results, setResults] = useState<Record<string, AnalysisResult>>({});
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [availableReports, setAvailableReports] = useState<Report[]>([]);
  const [downloadingReports, setDownloadingReports] = useState<Set<string>>(new Set());
  
  // Runtime clock state
  const [startTime, setStartTime] = useState<Date | null>(null);
  const [elapsedTime, setElapsedTime] = useState<number>(0); // in seconds
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Simple polling interval
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Helper function to format elapsed time
  const formatElapsedTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Set tab title and favicon
  useEffect(() => {
    // Title
    document.title = 'Kuwait Banking Analysis';

    // Favicon
    const existing = document.querySelector("link[rel='icon']") as HTMLLinkElement | null;
    const link = existing ?? document.createElement('link');
    link.rel = 'icon';
    link.href = '/gb-favicon.png';
    if (!existing) document.head.appendChild(link);
  }, []);

  // Runtime clock timer
  useEffect(() => {
    if (isRunning && startTime) {
      timerRef.current = setInterval(() => {
        const now = new Date();
        const elapsed = Math.floor((now.getTime() - startTime.getTime()) / 1000);
        setElapsedTime(elapsed);
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isRunning, startTime]);

  // Calculate total combinations
  useEffect(() => {
    const total = selectedBanks.size * selectedNewspapers.size;
    setProgress(prev => ({ ...prev, total }));
  }, [selectedBanks, selectedNewspapers]);

  // Simple polling function
  const pollProgress = async () => {
    if (!currentTaskId) return;

    try {
      const response = await fetch(`http://localhost:8000/api/progress/${currentTaskId}`);
      if (response.ok) {
        const data = await response.json();
        
        // Update progress
        if (data.progress) {
          setProgress({
            current: data.progress.current || 0,
            total: data.progress.total || 0,
            currentTask: data.progress.current_task || '',
            phase: data.progress.phase || 'idle'
          });
        }

        // Update results
        if (data.results) {
          setResults(data.results);
        }

        // Check if completed
        if (data.status === 'completed') {
          setIsRunning(false);
          setStartTime(null); // Stop the timer
          setProgress(prev => ({ ...prev, phase: 'complete' }));
          await fetchAvailableReports(currentTaskId);
          
          // Stop polling
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
        } else if (data.status === 'error') {
          setIsRunning(false);
          setStartTime(null); // Stop the timer on error
          setProgress(prev => ({ ...prev, phase: 'error' }));
          
          // Stop polling
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
        }
      }
    } catch (error) {
      console.error('Polling error:', error);
    }
  };

  // Start/stop polling based on running state
  useEffect(() => {
    if (isRunning && currentTaskId) {
      pollingRef.current = setInterval(pollProgress, 2000); // Poll every 2 seconds
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [isRunning, currentTaskId]);

  // Fetch available reports
  const fetchAvailableReports = async (taskId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/list-reports/${taskId}`);
      if (response.ok) {
        const data = await response.json();
        setAvailableReports(data.reports || []);
      }
    } catch (error) {
      console.error('Error fetching reports:', error);
    }
  };

  // Selection handlers
  const toggleBank = (bankId: string) => {
    const newSelected = new Set(selectedBanks);
    if (newSelected.has(bankId)) {
      newSelected.delete(bankId);
    } else {
      newSelected.add(bankId);
    }
    setSelectedBanks(newSelected);
  };

  const toggleNewspaper = (newspaperId: string) => {
    const newSelected = new Set(selectedNewspapers);
    if (newSelected.has(newspaperId)) {
      newSelected.delete(newspaperId);
    } else {
      newSelected.add(newspaperId);
    }
    setSelectedNewspapers(newSelected);
  };

  const selectAllBanks = () => setSelectedBanks(new Set(availableBanks.map(b => b.id)));
  const deselectAllBanks = () => setSelectedBanks(new Set());
  const selectAllNewspapers = () => setSelectedNewspapers(new Set(availableNewspapers.map(n => n.id)));
  const deselectAllNewspapers = () => setSelectedNewspapers(new Set());

  // Start analysis
  const runAnalysis = async () => {
    if (selectedBanks.size === 0 || selectedNewspapers.size === 0) {
      alert('Please select at least one bank and one newspaper.');
      return;
    }

    setIsRunning(true);
    setResults({});
    setAvailableReports([]);
    setStartTime(new Date()); // Start the timer
    setElapsedTime(0); // Reset elapsed time
    setProgress({
      current: 0,
      total: selectedBanks.size * selectedNewspapers.size,
      currentTask: 'Starting analysis...',
      phase: 'analyzing'
    });

    try {
      const response = await fetch('http://localhost:8000/api/start-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selected_banks: Array.from(selectedBanks),
          selected_newspapers: Array.from(selectedNewspapers),
          selected_model: selectedModel
          //selected_model: selectedModel
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setCurrentTaskId(data.task_id);
      console.log('Analysis started with task ID:', data.task_id);

    } catch (error) {
      console.error('Error starting analysis:', error);
      setIsRunning(false);
      setStartTime(null); // Stop timer on error
      alert('Failed to start analysis. Please check if the server is running.');
    }
  };

  // Stop analysis
  const stopAnalysis = async () => {
    setIsRunning(false);
    setStartTime(null); // Stop the timer
    
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    
    if (currentTaskId) {
      try {
        await fetch(`http://localhost:8000/api/tasks/${currentTaskId}`, { 
          method: 'DELETE' 
        });
      } catch (error) {
        console.error('Error cancelling task:', error);
      }
    }
    
    setCurrentTaskId(null);
  };

  // Download functions
  const downloadReport = async (report: Report) => {
    setDownloadingReports(prev => new Set(prev).add(report.type));
    try {
      const res = await fetch(`http://localhost:8000${report.download_url}`, {
        // credentials: 'include', // only if you later add cookies/auth
        mode: 'cors',
      });
  
      if (!res.ok) {
        // read server error body for real cause (e.g., "Analysis not completed", "Report not found")
        const text = await res.text().catch(() => '');
        throw new Error(`Download failed (${res.status}) ${text}`);
      }
  
      // Optional: use filename from header if present
      let filename = report.filename || report.name + '.docx';
      const disp = res.headers.get('Content-Disposition');
      const m = disp?.match(/filename="?([^"]+)"?/i);
      if (m && m[1]) filename = m[1];
  
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e: any) {
      console.error('Download error:', e);
      alert(`Failed to download ${report.name}\n${e?.message ?? ''}`);
    } finally {
      setDownloadingReports(prev => {
        const ns = new Set(prev);
        ns.delete(report.type);
        return ns;
      });
    }
  };

  const downloadAllReportsZip = async () => {
    if (!currentTaskId) return;

    setDownloadingReports(prev => new Set(prev).add('all_zip'));

    try {
      const response = await fetch(`http://localhost:8000/api/download-reports/${currentTaskId}`, {
        method: 'POST'
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `banking_analysis_${new Date().toISOString().split('T')[0]}.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading ZIP:', error);
      alert('Failed to download reports ZIP file');
    } finally {
      setDownloadingReports(prev => {
        const newSet = new Set(prev);
        newSet.delete('all_zip');
        return newSet;
      });
    }
  };

  // Get phase icon and color
  const getPhaseIcon = (phase: string) => {
    switch (phase) {
      case 'downloading': return <Download size={16} />;
      case 'analyzing': return <Eye size={16} />;
      case 'generating': return <FileText size={16} />;
      case 'complete': return <CheckCircle size={16} />;
      case 'error': return <AlertCircle size={16} />;
      default: return <Clock size={16} />;
    }
  };

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'downloading': return '#3b82f6';
      case 'analyzing': return '#8b5cf6';
      case 'generating': return '#10b981';
      case 'complete': return '#10b981';
      case 'error': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <div className="logo-container">
              <img
                src="/gbk_logo.png"
                alt="GBK Logo"
                style={{width: 280, height: 'auto', objectFit: 'contain',
                  background: 'white',padding: '4px 8px',borderRadius: '8px'
                }}
              />
              {isRunning && <div className="logo-pulse"></div>}
            </div>
            <div>
              <h1 className="header-title">Kuwait Banking Analysis</h1>
              <p className="header-subtitle">Real-time Multi-Bank Analysis System</p>
            </div>

                  {/* AI Model Selection - ADD THIS HERE */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              marginLeft: '2rem',
              padding: '0.5rem 1rem',
              background: 'rgba(255,255,255,0.1)',
              borderRadius: '0.5rem',
              border: '1px solid rgba(255,255,255,0.2)'
            }}>
              <Settings size={16} color="white" />
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value as 'gpt-4o' | 'gpt-4o-mini')}
                disabled={isRunning}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'white',
                  fontSize: '0.9rem',
                  fontWeight: '500',
                  cursor: 'pointer',
                  outline: 'none'
                }}
              >
                <option value="gpt-4o" style={{color: 'black'}}>GPT-4o</option>
                <option value="gpt-4o-mini" style={{color: 'black'}}>GPT-4o Mini</option>
              </select>
            </div>
          </div>
          
          <div className="header-right">
            <div className="date-info">
              <div className="date-text">
                <Calendar size={16} />
                <span>{new Date().toLocaleDateString()}</span>
              </div>
              {/* Runtime Clock */}
              <div className="runtime-clock" style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.5rem 1rem',
                background: isRunning ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)' : '#f3f4f6',
                color: isRunning ? 'white' : '#374151',
                borderRadius: '0.5rem',
                border: isRunning ? 'none' : '1px solid #d1d5db',
                fontFamily: 'monospace',
                fontSize: '1rem',
                fontWeight: '600',
                transition: 'all 0.3s ease',
                minWidth: '80px',
                justifyContent: 'center'
              }}>
                <Timer size={16} />
                <span>{formatElapsedTime(elapsedTime)}</span>
              </div>
              <div className="live-text">Live Analysis</div>
            </div>
          </div>
        </div>
      </header>

      {/* Status Bar - PROMINENT POSITION */}
      {isRunning && (
        <div style={{
          position: 'sticky',
          top: 0,
          zIndex: 100,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          padding: '1rem 2rem',
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            maxWidth: '1200px',
            margin: '0 auto'
          }}>
            <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
              <div style={{color: getPhaseColor(progress.phase)}}>
                {getPhaseIcon(progress.phase)}
              </div>
              <div>
                <div style={{fontSize: '1.1rem', fontWeight: '600', marginBottom: '0.25rem'}}>
                  {progress.currentTask || 'Processing...'}
                </div>
                <div style={{fontSize: '0.9rem', opacity: 0.9}}>
                  Progress: {progress.current} / {progress.total} combinations
                </div>
              </div>
            </div>
            
            <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
              {/* Runtime Clock in Status Bar */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.5rem 1rem',
                background: 'rgba(255,255,255,0.2)',
                borderRadius: '6px',
                fontFamily: 'monospace',
                fontSize: '1.1rem',
                fontWeight: '600'
              }}>
                <Timer size={18} />
                <span>{formatElapsedTime(elapsedTime)}</span>
              </div>
              
              <div style={{minWidth: '200px'}}>
                <div style={{
                  background: 'rgba(255,255,255,0.2)',
                  borderRadius: '10px',
                  height: '8px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    background: 'white',
                    height: '100%',
                    width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%`,
                    transition: 'width 0.3s ease'
                  }}></div>
                </div>
                <div style={{fontSize: '0.8rem', textAlign: 'center', marginTop: '0.25rem'}}>
                  {progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0}%
                </div>
              </div>
              
              <button 
                onClick={stopAnalysis}
                style={{
                  background: 'rgba(255,255,255,0.2)',
                  border: '1px solid rgba(255,255,255,0.3)',
                  borderRadius: '6px',
                  color: 'white',
                  padding: '0.5rem 1rem',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <X size={16} />
                Stop
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Container */}
      <div className="main-container">
        <div className="dashboard-grid">
          
          {/* Left Panel - Configuration */}
          <div className="config-panel">
            
            {/* Bank Selection */}
            <div className="selection-card">
              <div className="selection-header">
                <h2 className="card-title">
                  <div className="title-icon blue">
                    <Building2 size={20} />
                  </div>
                  Select Banks
                </h2>
                <span className="counter" style={{background: '#eff6ff', color: '#1d4ed8'}}>
                  {selectedBanks.size} selected
                </span>
              </div>
              
              <div className="selection-controls">
                <button className="control-btn blue" onClick={selectAllBanks}>Select All</button>
                <button className="control-btn gray" onClick={deselectAllBanks}>Clear All</button>
              </div>
              
              <div className="selection-list">
                {availableBanks.map(bank => (
                  <label key={bank.id} className="selection-item">
                    <input
                      type="checkbox"
                      className="checkbox"
                      checked={selectedBanks.has(bank.id)}
                      onChange={() => toggleBank(bank.id)}
                      disabled={isRunning}
                    />
                    <div className="item-content">
                      <div className="item-name">{bank.name}</div>
                      <div className="item-arabic">{bank.arabicName}</div>
                    </div>
                    {selectedBanks.has(bank.id) && (
                      <CheckCircle size={16} color="#2563eb" />
                    )}
                  </label>
                ))}
              </div>
            </div>

            {/* Newspaper Selection */}
            <div className="selection-card">
              <div className="selection-header">
                <h2 className="card-title">
                  <div className="title-icon green">
                    <Newspaper size={20} />
                  </div>
                  Select Newspapers
                </h2>
                <span className="counter" style={{background: '#ecfdf5', color: '#047857'}}>
                  {selectedNewspapers.size} selected
                </span>
              </div>
              
              <div className="selection-controls">
                <button className="control-btn green" onClick={selectAllNewspapers}>Select All</button>
                <button className="control-btn gray" onClick={deselectAllNewspapers}>Clear All</button>
              </div>
              
              <div className="selection-list">
                {availableNewspapers.map(newspaper => (
                  <label key={newspaper.id} className="selection-item">
                    <input
                      type="checkbox"
                      className="checkbox"
                      checked={selectedNewspapers.has(newspaper.id)}
                      onChange={() => toggleNewspaper(newspaper.id)}
                      disabled={isRunning}
                      style={{accentColor: '#059669'}}
                    />
                    <div className="item-content">
                      <div className="item-name">{newspaper.name}</div>
                      <div className="item-arabic">{newspaper.arabicName}</div>
                    </div>
                    {selectedNewspapers.has(newspaper.id) && (
                      <CheckCircle size={16} color="#059669" />
                    )}
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Right Panel - Main Analysis */}
          <div className="main-panel">
            
            {/* Control Panel */}
            <div className="control-card">
              <h2 className="card-title">
                <div className="title-icon purple">
                  <Zap size={20} />
                </div>
                Analysis Control
              </h2>
              
              <div className="control-section">
                <div className="control-buttons">
                  {!isRunning ? (
                    <button 
                      className="btn-primary"
                      onClick={runAnalysis}
                      disabled={selectedBanks.size === 0 || selectedNewspapers.size === 0}
                    >
                      <Play size={20} />
                      Start Analysis
                    </button>
                  ) : (
                    <div style={{
                      padding: '1rem',
                      background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
                      borderRadius: '0.75rem',
                      border: '1px solid #bfdbfe',
                      textAlign: 'center'
                    }}>
                      <div style={{fontSize: '1rem', fontWeight: '600', color: '#1d4ed8', marginBottom: '0.5rem'}}>
                        Analysis Running...
                      </div>
                      <div style={{fontSize: '0.875rem', color: '#3730a3'}}>
                        Check the status bar above for real-time progress
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Analysis Info */}
                <div className="progress-warning">
                  <AlertTriangle size={16} />
                  <span>
                  Analysis involves downloading PDFs and processing content with AI. Please be patient as each newspaper requires thorough examination.
                  </span>
                </div>
              </div>
            </div>

            {/* Reports Section */}
            {availableReports.length > 0 && (
              <div className="results-card">
                <h2 className="card-title">
                  <div className="title-icon green">
                    <FileText size={20} />
                  </div>
                  Generated Reports
                  <span className="counter" style={{background: '#ecfdf5', color: '#047857', marginLeft: '0.75rem'}}>
                    {availableReports.length} Available
                  </span>
                </h2>
                
                <div className="reports-section">
                  <div className="reports-grid">
                    {availableReports.map((report) => (
                      <div key={report.type} className="report-card">
                        <div className="report-icon">
                          {report.type === 'headlines' ? (
                            <Newspaper size={24} color="#2563eb" />
                          ) : (
                            <Building2 size={24} color="#059669" />
                          )}
                        </div>
                        <div className="report-info">
                          <div className="report-name">{report.name}</div>
                          <div className="report-filename">{report.filename}</div>
                        </div>
                        <button
                          className="report-download-btn"
                          onClick={() => downloadReport(report)}
                          disabled={downloadingReports.has(report.type)}
                        >
                          {downloadingReports.has(report.type) ? (
                            <RefreshCw size={16} className="animate-spin" />
                          ) : (
                            <Download size={16} />
                          )}
                        </button>
                      </div>
                    ))}
                  </div>
                  
                  <div className="reports-actions">
                    <button
                      className="btn-secondary"
                      onClick={downloadAllReportsZip}
                      disabled={downloadingReports.has('all_zip')}
                    >
                      {downloadingReports.has('all_zip') ? (
                        <RefreshCw size={16} className="animate-spin" />
                      ) : (
                        <Archive size={16} />
                      )}
                      {downloadingReports.has('all_zip') ? 'Preparing ZIP...' : 'Download All Reports as ZIP'}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Results Summary */}
            {Object.keys(results).length > 0 && (
              <div className="results-card">
                <h2 className="card-title">
                  <div className="title-icon green">
                    <CheckCircle size={20} />
                  </div>
                  Analysis Results
                  <span className="counter" style={{background: '#ecfdf5', color: '#047857', marginLeft: '0.75rem'}}>
                    {Object.keys(results).length} Completed
                  </span>
                </h2>
                
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                  gap: '1rem'
                }}>
                  {Object.entries(results).map(([key, result]) => (
                    <div key={key} style={{
                      padding: '1rem',
                      borderRadius: '0.75rem',
                      border: '1px solid #e2e8f0',
                      background: result.hasContent ? '#f0fdf4' : result.error ? '#fef2f2' : '#f9fafb'
                    }}>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: '0.5rem'
                      }}>
                        <div style={{fontWeight: '600', color: '#374151'}}>{result.bank}</div>
                        <div style={{fontSize: '0.875rem', color: '#6b7280'}}>{result.newspaper}</div>
                      </div>
                      
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        marginBottom: '0.5rem'
                      }}>
                        {result.hasContent ? (
                          <>
                            <CheckCircle size={16} color="#10b981" />
                            <span style={{color: '#059669', fontSize: '0.875rem'}}>
                              Content Found ({result.pages} pages)
                            </span>
                          </>
                        ) : result.error ? (
                          <>
                            <AlertCircle size={16} color="#ef4444" />
                            <span style={{color: '#dc2626', fontSize: '0.875rem'}}>Error</span>
                          </>
                        ) : (
                          <>
                            <Clock size={16} color="#6b7280" />
                            <span style={{color: '#6b7280', fontSize: '0.875rem'}}>No Content</span>
                          </>
                        )}
                      </div>
                      
                      {result.highlights && result.highlights.length > 0 && (
                        <div style={{fontSize: '0.8rem', color: '#4b5563'}}>
                          <div style={{fontWeight: '600', marginBottom: '0.25rem'}}>Headlines:</div>
                          {result.highlights.slice(0, 2).map((highlight, idx) => (
                            <div key={idx} style={{marginBottom: '0.125rem'}}>• {highlight}</div>
                          ))}
                          {result.highlights.length > 2 && (
                            <div style={{color: '#9ca3af'}}>+{result.highlights.length - 2} more...</div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;