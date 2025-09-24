import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Pause, Download, CheckCircle, AlertCircle, Clock, FileText, Building2, Newspaper, Users, TrendingUp, BarChart3, Settings, X, ArrowRight, Zap, Globe, Shield, Eye, RefreshCw, Calendar, AlertTriangle, File, Archive, Activity, Target, Award, Database } from 'lucide-react';
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

interface AnalysisStats {
  totalCombinations: number;
  completedCombinations: number;
  contentFound: number;
  errors: number;
}

interface TrackerStats {
  total_combinations: number;
  completed_combinations: number;
  combinations_with_content: number;
  total_banks: number;
  total_newspapers: number;
  banks_analyzed: number;
  newspapers_analyzed: number;
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
    { id: 'warba_bank', name: 'Warba Bank', arabicName: 'بنك وربة' }
  ];

  const availableNewspapers: NewspaperScript[] = [
    { id: 'alrai_multibank', name: 'Al-Rai Media', arabicName: 'الرأي' },
    { id: 'aljarida_multibank', name: 'Al-Jarida', arabicName: 'الجريدة' },
    { id: 'alqabas_multibank', name: 'Al-Qabas', arabicName: 'القبس' },
    { id: 'alnahar_multibank', name: 'Al-Nahar', arabicName: 'النهار' },
    { id: 'kwttimes_multibank', name: 'Kuwait Times', arabicName: 'كويت تايمز' },
    { id: 'arabtimes_multibank', name: 'Arab Times', arabicName: 'عرب تايمز' },
    { id: 'alwasat_multibank', name: 'Al-Wasat', arabicName: 'الوسط' },
    { id: 'alanbaa_multibanks', name: 'Al-Anbaa', arabicName: 'الأنباء' }
  ];

  // State management
  const [selectedBanks, setSelectedBanks] = useState<Set<string>>(new Set(availableBanks.map(b => b.id)));
  const [selectedNewspapers, setSelectedNewspapers] = useState<Set<string>>(new Set(availableNewspapers.map(n => n.id)));
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [progress, setProgress] = useState<Progress>({
    current: 0,
    total: 0,
    currentTask: '',
    phase: 'idle'
  });
  const [results, setResults] = useState<Record<string, AnalysisResult>>({});
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [analysisStats, setAnalysisStats] = useState<AnalysisStats>({
    totalCombinations: 0,
    completedCombinations: 0,
    contentFound: 0,
    errors: 0
  });
  
  const [trackerStats, setTrackerStats] = useState<TrackerStats>({
    total_combinations: 0,
    completed_combinations: 0,
    combinations_with_content: 0,
    total_banks: 0,
    total_newspapers: 0,
    banks_analyzed: 0,
    newspapers_analyzed: 0
  });
  const [methodology, setMethodology] = useState<string>('simple_realtime');
  const [currentConfig, setCurrentConfig] = useState<{banks: string[], newspapers: string[]}>({banks: [], newspapers: []});
  
  const [webSocket, setWebSocket] = useState<WebSocket | null>(null);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [availableReports, setAvailableReports] = useState<Report[]>([]);
  const [downloadingReports, setDownloadingReports] = useState<Set<string>>(new Set());

  // Enhanced WebSocket management state
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [lastUpdateTime, setLastUpdateTime] = useState<Date>(new Date());
  const maxReconnectAttempts = 5;
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const webSocketRef = useRef<WebSocket | null>(null);

  // State for manual checking and polling
  const [isManuallyChecking, setIsManuallyChecking] = useState(false);
  const [showManualCheck, setShowManualCheck] = useState(false);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Calculate total combinations when selections change
  useEffect(() => {
    const total = selectedBanks.size * selectedNewspapers.size;
    setProgress(prev => ({ ...prev, total }));
    setAnalysisStats(prev => ({ ...prev, totalCombinations: total }));
    setTrackerStats(prev => ({ ...prev, total_combinations: total }));
  }, [selectedBanks, selectedNewspapers]);

  // Reset analysis stats when starting new analysis
  useEffect(() => {
    if (isRunning) {
      const correctTotal = selectedBanks.size * selectedNewspapers.size;
      setAnalysisStats({
        totalCombinations: correctTotal,
        completedCombinations: 0,
        contentFound: 0,
        errors: 0
      });
      setTrackerStats({
        total_combinations: correctTotal,
        completed_combinations: 0,
        combinations_with_content: 0,
        total_banks: selectedBanks.size,
        total_newspapers: selectedNewspapers.size,
        banks_analyzed: 0,
        newspapers_analyzed: 0
      });
      setProgress(prev => ({ ...prev, total: correctTotal }));
    }
  }, [isRunning, selectedBanks.size, selectedNewspapers.size]);

  // Automatic polling as backup when WebSocket is disconnected
  useEffect(() => {
    if (currentTaskId && isRunning && (connectionStatus === 'disconnected' || connectionStatus === 'error')) {
      console.log('🔄 Starting polling backup due to WebSocket disconnection');
      
      pollingIntervalRef.current = setInterval(async () => {
        try {
          console.log('📡 Backup polling for task status...');
          const response = await fetch(`http://localhost:8000/api/progress/${currentTaskId}`);
          if (response.ok) {
            const data = await response.json();
            
            // Update progress even when WebSocket is down
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

            // Update tracker stats
            if (data.tracker_stats) {
              setTrackerStats(data.tracker_stats);
            }

            // Check if analysis completed while WebSocket was disconnected
            if (data.status === 'completed' && data.report_paths && availableReports.length === 0) {
              console.log('✅ Backup polling detected completed analysis - updating UI');
              setIsRunning(false);
              setProgress(prev => ({ ...prev, phase: 'complete' }));
              await fetchAvailableReports(currentTaskId);
              
              // Stop polling once we've detected completion
              if (pollingIntervalRef.current) {
                clearInterval(pollingIntervalRef.current);
                pollingIntervalRef.current = null;
              }
            }
          }
        } catch (error) {
          console.error('Backup polling error:', error);
        }
      }, 15000); // Poll every 15 seconds as backup
    }
    
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [currentTaskId, isRunning, connectionStatus, availableReports.length]);

  // Show manual check button when appropriate
  useEffect(() => {
    const shouldShowManualCheck = Boolean(
      currentTaskId && 
      !isRunning && 
      progress.phase !== 'complete' && 
      availableReports.length === 0 &&
      (connectionStatus === 'disconnected' || connectionStatus === 'error')
    );
    setShowManualCheck(shouldShowManualCheck);
  }, [currentTaskId, isRunning, progress.phase, availableReports.length, connectionStatus]);

  // Enhanced WebSocket cleanup function
  const cleanupWebSocket = useCallback(() => {
    if (webSocketRef.current) {
      webSocketRef.current.onopen = null;
      webSocketRef.current.onmessage = null;
      webSocketRef.current.onerror = null;
      webSocketRef.current.onclose = null;
      
      if (webSocketRef.current.readyState === WebSocket.OPEN) {
        webSocketRef.current.close();
      }
      webSocketRef.current = null;
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    
    setWebSocket(null);
    setConnectionStatus('disconnected');
  }, []);

  // Fetch available reports after analysis completion
  const fetchAvailableReports = async (taskId: string) => {
    try {
      console.log('📥 Fetching available reports for task:', taskId);
      const response = await fetch(`http://localhost:8000/api/list-reports/${taskId}`);
      if (response.ok) {
        const data = await response.json();
        console.log('📋 Available reports:', data.reports?.length || 0);
        setAvailableReports(data.reports || []);
      } else {
        console.error('Failed to fetch reports:', response.status);
      }
    } catch (error) {
      console.error('Error fetching reports:', error);
    }
  };

  // Manual check function for completed analysis
  const manuallyCheckForReports = async () => {
    if (!currentTaskId) return;

    setIsManuallyChecking(true);
    console.log('🔍 Manually checking for completed analysis and reports...');

    try {
      const response = await fetch(`http://localhost:8000/api/progress/${currentTaskId}`);
      if (response.ok) {
        const data = await response.json();
        console.log('📊 Task status:', data.status, 'Has reports:', !!data.report_paths);

        if (data.progress) {
          setProgress({
            current: data.progress.current || 0,
            total: data.progress.total || 0,
            currentTask: data.progress.current_task || '',
            phase: data.progress.phase || 'idle'
          });
        }

        if (data.results) {
          setResults(data.results);
        }

        if (data.tracker_stats) {
          setTrackerStats(data.tracker_stats);
        }

        if (data.status === 'completed' && data.report_paths) {
          console.log('✅ Found completed analysis with reports!');
          setIsRunning(false);
          setProgress(prev => ({ ...prev, phase: 'complete' }));
          await fetchAvailableReports(currentTaskId);
          setShowManualCheck(false);
          alert('✅ Analysis completed! Reports are now available for download.');
        } else if (data.status === 'completed' && !data.report_paths) {
          console.log('⚠️ Analysis completed but no reports found');
          alert('Analysis is marked as completed but reports are not yet available. Please wait a moment and try again.');
        } else if (data.status === 'running') {
          console.log('🔄 Analysis still running...');
          alert('Analysis is still running. Please wait for completion or try again later.');
        } else if (data.status === 'error') {
          console.log('❌ Analysis failed');
          setIsRunning(false);
          setProgress(prev => ({ ...prev, phase: 'error' }));
          alert('Analysis failed. Please check the logs or start a new analysis.');
        } else {
          console.log('❓ Unknown status:', data.status);
          alert(`Analysis status: ${data.status}. Reports may not be ready yet.`);
        }
      } else {
        console.error('Failed to fetch task status:', response.status);
        alert('Failed to check task status. Please ensure the server is running.');
      }
    } catch (error) {
      console.error('Error checking for reports:', error);
      alert('Error occurred while checking for reports. Please try again.');
    } finally {
      setIsManuallyChecking(false);
    }
  };

  // SIMPLE & RELIABLE: Enhanced WebSocket handling for file-based progress tracking
  const connectWebSocket = useCallback((taskId: string) => {
    if (webSocketRef.current?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    setConnectionStatus('connecting');
    
    const ws = new WebSocket(`ws://localhost:8000/ws/analysis/${taskId}`);
    webSocketRef.current = ws;

    ws.onopen = () => {
      console.log('🔌 WebSocket connected with simple progress monitoring');
      setConnectionStatus('connected');
      setReconnectAttempts(0);
      setWebSocket(ws);
    };

    ws.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data);
        setLastUpdateTime(new Date());
        
        // SIMPLE: Handle file-based progress updates
        if (update.status === 'progress_update' && update.file_progress) {
          const fileProgress = update.file_progress;
          console.log('📊 File progress update:', fileProgress);
          
          // Update current task description
          const currentTask = `${fileProgress.current_newspaper || 'Processing'}: ${fileProgress.current_bank || 'Working...'}`;
          
          setProgress(prev => ({
            ...prev,
            currentTask: currentTask,
            phase: fileProgress.current_phase || 'analyzing'
          }));
          
          // Update statistics from file
          if (fileProgress.completed_combinations !== undefined) {
            setProgress(prev => ({
              ...prev,
              current: fileProgress.completed_combinations
            }));
          }
          
          // Update analysis stats
          setAnalysisStats(prev => ({
            ...prev,
            completedCombinations: fileProgress.completed_combinations || prev.completedCombinations,
            contentFound: fileProgress.banks_with_content || prev.contentFound,
            errors: fileProgress.errors || prev.errors
          }));
          
          // Update tracker stats
          setTrackerStats(prev => ({
            ...prev,
            completed_combinations: fileProgress.completed_combinations || prev.completed_combinations,
            combinations_with_content: fileProgress.banks_with_content || prev.combinations_with_content,
            newspapers_analyzed: fileProgress.newspapers_completed || prev.newspapers_analyzed
          }));
          
          // Show estimation if available
          if (fileProgress.estimated_progress) {
            console.log(`🎯 Estimated progress: ${fileProgress.estimated_progress.toFixed(1)}%`);
          }
          
          return; // Exit early for file progress updates
        }
        
        // Handle standard WebSocket messages
        switch (update.status) {
          case 'not_found':
            console.error('Task not found');
            setIsRunning(false);
            setConnectionStatus('error');
            return;

          case 'connected':
            console.log('Simple progress monitoring confirmed:', update.message);
            setMethodology('simple_realtime');
            return;

          case 'heartbeat':
            console.log('Heartbeat received');
            return;

          case 'reports_available':
            console.log('🎯 Reports available');
            setIsRunning(false);
            setProgress(prev => ({ ...prev, phase: 'complete' }));
            fetchAvailableReports(taskId);
            return;

          default:
            // Handle regular progress updates (backup mechanism)
            if (update.progress) {
              setProgress(prevProgress => {
                const newProgress = {
                  current: update.progress.current || prevProgress.current,
                  total: update.progress.total || prevProgress.total,
                  currentTask: update.progress.current_task || prevProgress.currentTask,
                  phase: update.progress.phase || prevProgress.phase
                };
                
                console.log('📊 Standard progress update:', newProgress);
                return newProgress;
              });
            }

            // Handle results updates
            if (update.results) {
              setResults(prevResults => {
                const newResults = { ...prevResults, ...update.results };
                
                // Calculate stats from results
                const contentFound = Object.values(newResults).filter(
                  (result: any) => result.hasContent
                ).length;
                const errors = Object.values(newResults).filter(
                  (result: any) => result.error
                ).length;
                const completedCombinations = Object.keys(newResults).length;
                
                // Update analysis stats
                setAnalysisStats(prev => ({
                  totalCombinations: prev.totalCombinations,
                  completedCombinations,
                  contentFound,
                  errors
                }));
                
                return newResults;
              });
            }

            // Handle tracker stats updates
            if (update.tracker_stats) {
              setTrackerStats(prevStats => ({
                ...prevStats,
                ...update.tracker_stats
              }));
            }

            if (update.methodology) {
              setMethodology(update.methodology);
            }
            if (update.config) {
              setCurrentConfig(update.config);
            }

            if (update.status === 'completed') {
              console.log('✅ Analysis completed via WebSocket');
              setIsRunning(false);
              setProgress(prev => ({ ...prev, phase: 'complete' }));
              fetchAvailableReports(taskId);
              setTimeout(() => cleanupWebSocket(), 5000);
              
            } else if (update.status === 'error') {
              setIsRunning(false);
              setProgress(prev => ({ ...prev, phase: 'error' }));
              setConnectionStatus('error');
              console.error('Analysis failed:', update.error);
              alert(`Analysis failed: ${update.error || 'Unknown error'}`);
            }
            break;
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };

    ws.onclose = (event) => {
      console.log('WebSocket connection closed:', event.code, event.reason);
      setConnectionStatus('disconnected');
      webSocketRef.current = null;
      setWebSocket(null);

      if (isRunning && reconnectAttempts < maxReconnectAttempts && event.code !== 1000) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
        console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`);
        
        reconnectTimeoutRef.current = setTimeout(() => {
          setReconnectAttempts(prev => prev + 1);
          connectWebSocket(taskId);
        }, delay);
      } else if (reconnectAttempts >= maxReconnectAttempts) {
        console.error('Max reconnection attempts reached');
        setIsRunning(false);
        alert('Connection lost. The analysis may still be running in the background. Use the "Check for Completed Reports" button to verify completion.');
      }
    };

  }, [isRunning, reconnectAttempts, cleanupWebSocket]);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      cleanupWebSocket();
    };
  }, [cleanupWebSocket]);

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

  // Enhanced analysis process
  const runAnalysis = async () => {
    if (selectedBanks.size === 0 || selectedNewspapers.size === 0) {
      alert('Please select at least one bank and one newspaper.');
      return;
    }

    setIsRunning(true);
    setIsPaused(false);
    setResults({});
    setAvailableReports([]);
    setReconnectAttempts(0);
    setShowManualCheck(false);
    setAnalysisStats({
      totalCombinations: selectedBanks.size * selectedNewspapers.size,
      completedCombinations: 0,
      contentFound: 0,
      errors: 0
    });
    setTrackerStats({
      total_combinations: selectedBanks.size * selectedNewspapers.size,
      completed_combinations: 0,
      combinations_with_content: 0,
      total_banks: selectedBanks.size,
      total_newspapers: selectedNewspapers.size,
      banks_analyzed: 0,
      newspapers_analyzed: 0
    });

    cleanupWebSocket();

    try {
      const response = await fetch('http://localhost:8000/api/start-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selected_banks: Array.from(selectedBanks),
          selected_newspapers: Array.from(selectedNewspapers)
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data = await response.json();
      const taskId = data.task_id;
      setCurrentTaskId(taskId);
      setMethodology(data.methodology || 'simple_realtime');

      console.log('Simple real-time analysis started with task ID:', taskId);
      console.log('Using methodology:', data.methodology);

      connectWebSocket(taskId);

    } catch (error) {
      console.error('Error starting simple real-time analysis:', error);
      setIsRunning(false);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      alert(`Failed to start analysis: ${errorMessage}. Please check if the server is running.`);
    }
  };

  // Download individual report
  const downloadReport = async (report: Report) => {
    if (!currentTaskId) return;

    setDownloadingReports(prev => new Set(prev).add(report.type));

    try {
      const response = await fetch(`http://localhost:8000${report.download_url}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = report.filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

    } catch (error) {
      console.error('Error downloading report:', error);
      alert(`Failed to download ${report.name}`);
    } finally {
      setDownloadingReports(prev => {
        const newSet = new Set(prev);
        newSet.delete(report.type);
        return newSet;
      });
    }
  };

  // Download all reports as ZIP
  const downloadAllReportsZip = async () => {
    if (!currentTaskId) return;

    setDownloadingReports(prev => new Set(prev).add('all_zip'));

    try {
      const response = await fetch(`http://localhost:8000/api/download-reports/${currentTaskId}`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `simple_realtime_banking_analysis_${new Date().toISOString().split('T')[0]}.zip`;
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

  const pauseAnalysis = () => {
    setIsPaused(true);
  };

  const resumeAnalysis = () => {
    setIsPaused(false);
  };

  // Function to aggregate results by bank for the redesigned results section
  const aggregateResultsByBank = () => {
    const bankSummaries: Record<string, {
      bankName: string;
      bankId: string;
      totalNewspapers: number;
      newspapersWithContent: number;
      totalPages: number;
      allHighlights: string[];
      hasContent: boolean;
      errorCount: number;
    }> = {};

    // Initialize summaries for all selected banks
    availableBanks.forEach(bank => {
      if (selectedBanks.has(bank.id)) {
        bankSummaries[bank.id] = {
          bankName: bank.name,
          bankId: bank.id,
          totalNewspapers: 0,
          newspapersWithContent: 0,
          totalPages: 0,
          allHighlights: [],
          hasContent: false,
          errorCount: 0
        };
      }
    });

    // Aggregate results
    Object.entries(results).forEach(([key, result]) => {
      // Extract bank ID from the result key (format: bankId_newspaperId)
      const bankId = key.split('_')[0];
      
      if (bankSummaries[bankId]) {
        bankSummaries[bankId].totalNewspapers++;
        
        if (result.hasContent) {
          bankSummaries[bankId].newspapersWithContent++;
          bankSummaries[bankId].totalPages += result.pages;
          bankSummaries[bankId].allHighlights.push(...result.highlights);
          bankSummaries[bankId].hasContent = true;
        }
        
        if (result.error) {
          bankSummaries[bankId].errorCount++;
        }
      }
    });

    return Object.values(bankSummaries).filter(summary => summary.totalNewspapers > 0);
  };

// Enhanced stop function with proper cleanup
const stopAnalysis = async () => {
  setIsRunning(false);
  setIsPaused(false);
  setShowManualCheck(false);
  
  cleanupWebSocket();
  
  if (currentTaskId) {
    try {
      await fetch(`http://localhost:8000/api/tasks/${currentTaskId}`, { 
        method: 'DELETE' 
      });
      console.log('Simple real-time task cancelled on backend');
    } catch (error) {
      console.error('Error cancelling task:', error);
    }
  }
  
  setCurrentTaskId(null);
};

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
    case 'downloading': return 'blue';
    case 'analyzing': return 'purple';
    case 'generating': return 'green';
    case 'complete': return 'green';
    case 'error': return 'red';
    default: return 'gray';
  }
};

// SIMPLE: Enhanced connection status indicator
const ConnectionStatusIndicator = () => {
  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Simple Real-time Connected';
      case 'connecting': return 'Connecting...';
      case 'error': return 'Connection Error (Fallback Active)';
      default: return 'Disconnected (Fallback Active)';
    }
  };

  return (
    <div className="status-indicator">
      <div className={`status-dot ${connectionStatus}`}></div>
      <span>{getStatusText()}</span>
      {reconnectAttempts > 0 && (
        <span style={{fontSize: '0.75rem', color: '#ea580c'}}>
          (Retry {reconnectAttempts}/{maxReconnectAttempts})
        </span>
      )}
      {connectionStatus === 'connected' && (
        <span style={{fontSize: '0.75rem', color: '#059669'}}>
          (File-based progress)
        </span>
      )}
      {(connectionStatus === 'disconnected' || connectionStatus === 'error') && isRunning && (
        <span style={{fontSize: '0.75rem', color: '#2563eb'}}>
          (Backup polling active)
        </span>
      )}
    </div>
  );
};

// SIMPLE: Enhanced progress display with file-based progress
const ProgressDisplay = () => {
  return (
    <div className="progress-section">
      <div className="progress-header">
        <div className={`progress-phase ${getPhaseColor(progress.phase)}`}>
          {getPhaseIcon(progress.phase)}
          <span className="progress-phase-text">
            {progress.phase === 'generating' ? 'Generating Reports' : progress.phase}
          </span>
          {methodology === 'simple_realtime' && (
            <span className="counter" style={{background: '#f0f9ff', color: '#0369a1'}}>
              SIMPLE
            </span>
          )}
        </div>
        <span className="progress-counter">
          {progress.current} / {progress.total}
        </span>
      </div>
      
      <div className="progress-bar">
        <div 
          className="progress-fill"
          style={{ 
            width: `${Math.min(100, (progress.current / progress.total) * 100)}%`,
            transition: 'width 0.5s ease-in-out'
          }}
        ></div>
      </div>
      
      <div className="progress-task" style={{
        minHeight: '1.5rem',
        transition: 'all 0.3s ease-in-out',
        fontSize: '0.875rem',
        color: '#374151'
      }}>
        {progress.currentTask || 'Processing...'}
      </div>
      
      {/* Simple Real-time Indicator */}
      {connectionStatus === 'connected' && isRunning && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          fontSize: '0.75rem',
          color: '#0369a1',
          marginTop: '0.5rem'
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: '#0369a1',
            animation: 'pulse 2s infinite'
          }}></div>
          Simple real-time tracking active
        </div>
      )}
      
      {/* Analysis Time Warning */}
      <div className="progress-warning">
        <AlertTriangle size={16} />
        <span>
          Simple real-time analysis with file-based progress tracking. Updates every few seconds.
        </span>
      </div>

      {/* Simple Real-time Progress Metrics */}
      {methodology === 'simple_realtime' && analysisStats.completedCombinations > 0 && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '0.75rem',
          padding: '1rem',
          background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
          borderRadius: '0.75rem',
          border: '1px solid #bae6fd'
        }}>
          <div style={{textAlign: 'center'}}>
            <div style={{
              fontSize: '1.125rem', 
              fontWeight: '700', 
              color: '#0369a1',
              transition: 'all 0.5s ease-in-out'
            }}>
              {analysisStats.completedCombinations}
            </div>
            <div style={{fontSize: '0.75rem', color: '#6b7280'}}>Completed</div>
          </div>
          <div style={{textAlign: 'center'}}>
            <div style={{
              fontSize: '1.125rem', 
              fontWeight: '700', 
              color: '#059669',
              transition: 'all 0.5s ease-in-out'
            }}>
              {analysisStats.contentFound}
            </div>
            <div style={{fontSize: '0.75rem', color: '#6b7280'}}>With Content</div>
          </div>
          <div style={{textAlign: 'center'}}>
            <div style={{
              fontSize: '1.125rem', 
              fontWeight: '700', 
              color: '#dc2626',
              transition: 'all 0.5s ease-in-out'
            }}>
              {analysisStats.errors}
            </div>
            <div style={{fontSize: '0.75rem', color: '#6b7280'}}>Errors</div>
          </div>
          <div style={{textAlign: 'center'}}>
            <div style={{
              fontSize: '1.125rem', 
              fontWeight: '700', 
              color: '#2563eb',
              transition: 'all 0.5s ease-in-out'
            }}>
              {analysisStats.completedCombinations > 0 
                ? Math.round((analysisStats.contentFound / analysisStats.completedCombinations) * 100)
                : 0}%
            </div>
            <div style={{fontSize: '0.75rem', color: '#6b7280'}}>Success Rate</div>
          </div>
        </div>
      )}
    </div>
  );
};

return (
  <div className="app-container">
    {/* Header */}
    <header className="header">
      <div className="header-content">
        <div className="header-left">
          <div className="logo-container">
            <div className="logo-icon">
              <Building2 size={32} color="white" />
            </div>
            {isRunning && <div className="logo-pulse"></div>}
          </div>
          <div>
            <h1 className="header-title">Kuwait Banking News Analyzer</h1>
            <p className="header-subtitle">
              Real-time Multi-Bank Analysis System • {methodology === 'simple_realtime' ? 'Simple Real-time' : 'Standard'} Mode
            </p>
          </div>
        </div>
        
        <div className="header-right">
          <ConnectionStatusIndicator />
          <div className="status-indicator">
            <div className="status-dot"></div>
            <span>System Online</span>
          </div>
          <button 
            className="settings-btn"
            onClick={() => setShowSettings(!showSettings)}
          >
            <Settings size={20} />
          </button>
          <div className="date-info">
            <div className="date-text">
              <Calendar size={16} />
              <span>{new Date().toLocaleDateString()}</span>
            </div>
            <div className="live-text">Live Analysis</div>
          </div>
        </div>
      </div>
    </header>

    {/* Main Container */}
    <div className="main-container">
      <div className="dashboard-grid">
        
        {/* Left Panel - Configuration */}
        <div className="config-panel">
          
          {/* Enhanced Statistics Card with Simple Real-time Data */}
          <div className="stats-card">
            <div className="selection-header">
              <h2 className="card-title">
                <div className="title-icon blue">
                  <BarChart3 size={20} />
                </div>
                Analysis Statistics
              </h2>
              {methodology === 'simple_realtime' && (
                <span className="counter" style={{background: '#f0f9ff', color: '#0369a1'}}>
                  Simple Real-time
                </span>
              )}
            </div>
            
            <div className="stats-grid">
              <div className="stat-item blue">
                <div className="stat-number">{trackerStats.total_combinations}</div>
                <div className="stat-label">Total Tasks</div>
              </div>
              <div className="stat-item green">
                <div className="stat-number">{trackerStats.combinations_with_content}</div>
                <div className="stat-label">Content Found</div>
              </div>
              <div className="stat-item purple">
                <div className="stat-number">{trackerStats.completed_combinations}</div>
                <div className="stat-label">Completed</div>
              </div>
              <div className="stat-item orange">
                <div className="stat-number">{analysisStats.errors}</div>
                <div className="stat-label">Errors</div>
              </div>
            </div>

            {/* Simple Real-time Enhanced Stats */}
            {methodology === 'simple_realtime' && trackerStats.banks_analyzed > 0 && (
              <div style={{borderTop: '1px solid #e2e8f0', paddingTop: '1rem', marginTop: '1rem'}}>
                <h3 style={{fontSize: '0.875rem', fontWeight: '600', color: '#374151', marginBottom: '0.5rem', display: 'flex', alignItems: 'center'}}>
                  <Target size={16} style={{marginRight: '0.5rem'}} />
                  Simple Real-time Metrics
                </h3>
                <div style={{display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.875rem'}}>
                  <div style={{display: 'flex', justifyContent: 'space-between'}}>
                    <span style={{color: '#6b7280'}}>Banks Analyzed:</span>
                    <span style={{fontWeight: '500'}}>{trackerStats.banks_analyzed}/{trackerStats.total_banks}</span>
                  </div>
                  <div style={{display: 'flex', justifyContent: 'space-between'}}>
                    <span style={{color: '#6b7280'}}>Newspapers Analyzed:</span>
                    <span style={{fontWeight: '500'}}>{trackerStats.newspapers_analyzed}/{trackerStats.total_newspapers}</span>
                  </div>
                  <div style={{display: 'flex', justifyContent: 'space-between'}}>
                    <span style={{color: '#6b7280'}}>Success Rate:</span>
                    <span style={{fontWeight: '500'}}>
                      {trackerStats.total_combinations > 0 
                        ? Math.round((trackerStats.combinations_with_content / trackerStats.total_combinations) * 100)
                        : 0}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

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
              <button 
                className="control-btn blue"
                onClick={selectAllBanks}
              >
                Select All
              </button>
              <button 
                className="control-btn gray"
                onClick={deselectAllBanks}
              >
                Clear All
              </button>
            </div>
            
            <div className="selection-list">
              {availableBanks.map(bank => (
                <label key={bank.id} className="selection-item">
                  <input
                    type="checkbox"
                    className="checkbox"
                    checked={selectedBanks.has(bank.id)}
                    onChange={() => toggleBank(bank.id)}
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
              <button 
                className="control-btn green"
                onClick={selectAllNewspapers}
              >
                Select All
              </button>
              <button 
                className="control-btn gray"
                onClick={deselectAllNewspapers}
              >
                Clear All
              </button>
            </div>
            
            <div className="selection-list">
              {availableNewspapers.map(newspaper => (
                <label key={newspaper.id} className="selection-item">
                  <input
                    type="checkbox"
                    className="checkbox"
                    checked={selectedNewspapers.has(newspaper.id)}
                    onChange={() => toggleNewspaper(newspaper.id)}
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
              {methodology === 'simple_realtime' && (
                <span className="counter" style={{background: '#f0f9ff', color: '#0369a1', marginLeft: '0.75rem'}}>
                  Simple Real-time Engine
                </span>
              )}
            </h2>
            
            <div className="control-section">
              {/* Control Buttons */}
              <div className="control-buttons">
                {!isRunning ? (
                  <button 
                    className="btn-primary"
                    onClick={runAnalysis}
                    disabled={selectedBanks.size === 0 || selectedNewspapers.size === 0}
                  >
                    <Play size={20} />
                    Start Simple Real-time Analysis
                  </button>
                ) : (
                  <>
                    {!isPaused ? (
                      <button 
                        className="btn-warning"
                        onClick={pauseAnalysis}
                      >
                        <Pause size={20} />
                        Pause
                      </button>
                    ) : (
                      <button 
                        className="btn-success"
                        onClick={resumeAnalysis}
                      >
                        <Play size={20} />
                        Resume
                      </button>
                    )}
                    <button 
                      className="btn-danger"
                      onClick={stopAnalysis}
                    >
                      <X size={20} />
                      Stop
                    </button>
                  </>
                )}
              </div>

              {/* Manual Check Button */}
              {showManualCheck && (
                <div style={{
                  padding: '1rem',
                  background: 'linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)',
                  border: '1px solid #fde68a',
                  borderRadius: '0.75rem'
                }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: '0.75rem'
                  }}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                      <AlertTriangle size={20} color="#d97706" />
                      <span style={{fontSize: '0.875rem', fontWeight: '600', color: '#92400e'}}>
                        Connection Lost - Manual Check Available
                      </span>
                    </div>
                  </div>
                  <p style={{fontSize: '0.875rem', color: '#b45309', marginBottom: '0.75rem'}}>
                    The WebSocket connection was lost, but your analysis may have completed. 
                    Click below to check for finished reports.
                  </p>
                  <button 
                    className="btn-warning"
                    onClick={manuallyCheckForReports}
                    disabled={isManuallyChecking}
                    style={{width: '100%'}}
                  >
                    {isManuallyChecking ? (
                      <>
                        <RefreshCw size={16} className="animate-spin" />
                        Checking for Reports...
                      </>
                    ) : (
                      <>
                        <Eye size={16} />
                        Check for Completed Reports
                      </>
                    )}
                  </button>
                </div>
              )}
              
              {/* Enhanced Progress Section */}
              {isRunning && <ProgressDisplay />}
            </div>
          </div>

          {/* Enhanced Reports Section */}
          {availableReports.length > 0 && (
            <div className="results-card">
              <h2 className="card-title">
                <div className="title-icon green">
                  <FileText size={20} />
                </div>
                Generated Reports
                <span className="counter" style={{background: '#ecfdf5', color: '#047857', marginLeft: '0.75rem'}}>
                  Simple Real-time Generated
                </span>
              </h2>
              
              <div className="reports-section">
                {/* Individual Reports */}
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
                
                {/* Download All Option */}
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

          {/* Bank Summary Results Section */}
          {Object.keys(results).length > 0 && (
            <div className="results-card">
              <h2 className="card-title">
                <div className="title-icon green">
                  <CheckCircle size={20} />
                </div>
                Banking News Summary
                {methodology === 'simple_realtime' && (
                  <span className="counter" style={{background: '#f0f9ff', color: '#0369a1', marginLeft: '0.75rem'}}>
                    Simple Real-time Results
                  </span>
                )}
              </h2>
              
              <div className="results-grid">
                {aggregateResultsByBank().map((bankSummary) => (
                  <div key={bankSummary.bankId} className={`result-item ${
                    bankSummary.hasContent ? 'success' : bankSummary.errorCount > 0 ? 'error' : 'neutral'
                  }`}>
                    <div className="result-header">
                      <div className="result-bank">{bankSummary.bankName}</div>
                      <div className="result-newspaper">
                        {bankSummary.newspapersWithContent}/{bankSummary.totalNewspapers} newspapers with content
                      </div>
                    </div>
                    
                    <div className="result-status">
                      {bankSummary.hasContent ? (
                        <div className="status-success">
                          <CheckCircle size={20} />
                          Found in {bankSummary.newspapersWithContent} sources ({bankSummary.totalPages} pages)
                        </div>
                      ) : bankSummary.errorCount > 0 ? (
                        <div className="status-error">
                          <AlertCircle size={20} />
                          {bankSummary.errorCount} Analysis Errors
                        </div>
                      ) : (
                        <div className="status-neutral">
                          <Clock size={20} />
                          No Content Found
                        </div>
                      )}
                    </div>
                    
                    {bankSummary.allHighlights.length > 0 && (
                      <div className="result-highlights">
                        <div style={{fontWeight: '600', marginBottom: '0.5rem', color: '#374151'}}>
                          Major Headlines:
                        </div>
                        {bankSummary.allHighlights.slice(0, 4).map((highlight, idx) => (
                          <div key={idx} className="highlight-item">
                            • {highlight}
                          </div>
                        ))}
                        {bankSummary.allHighlights.length > 4 && (
                          <div style={{fontSize: '0.75rem', color: '#6b7280', marginLeft: '0.75rem', marginTop: '0.5rem'}}>
                            +{bankSummary.allHighlights.length - 4} more headlines across {bankSummary.newspapersWithContent} newspapers...
                          </div>
                        )}
                      </div>
                    )}
                    
                    {bankSummary.errorCount > 0 && !bankSummary.hasContent && (
                      <div style={{
                        marginTop: '0.5rem',
                        fontSize: '0.875rem',
                        color: '#dc2626',
                        background: '#fef2f2',
                        padding: '0.5rem',
                        borderRadius: '0.375rem'
                      }}>
                        {bankSummary.errorCount} newspapers failed to analyze
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
