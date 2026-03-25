import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const PharmaSalesApp());
}

class PharmaSalesApp extends StatelessWidget {
  const PharmaSalesApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pharma Sales Predictor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const PredictionPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();

  // Controllers
  final TextEditingController _yearController = TextEditingController();
  final TextEditingController _monthController = TextEditingController();
  final TextEditingController _weekdayController = TextEditingController();

  String _resultMessage = '';
  bool _isLoading = false;

  // IMPORTANT: For local android emulator, use 10.0.2.2.
  // For web/iOS simulator, use localhost.
  // Replace this URL with your Render URL for the submission!
  // e.g., 'https://my-api-service.onrender.com/predict'
  final String apiUrl = 'http://127.0.0.1:8000/predict';

  Future<void> _makePrediction() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoading = true;
      _resultMessage = '';
    });

    try {
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'year': int.parse(_yearController.text),
          'month': int.parse(_monthController.text),
          'weekday': int.parse(_weekdayController.text),
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _resultMessage = 'Predicted Sales: \ units';
        });
      } else {
        final err = jsonDecode(response.body);
        setState(() {
          _resultMessage = 'Error: \';
        });
      }
    } catch (e) {
      setState(() {
        _resultMessage = 'Connection Error: Failed to reach the API.\\n(Did you replace the apiUrl with your Render URL?)';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  void dispose() {
    _yearController.dispose();
    _monthController.dispose();
    _weekdayController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pharma Sales Predictor'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Enter Prediction Details',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              
              // Year Input
              TextFormField(
                controller: _yearController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Year (e.g., 2023)',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.calendar_today),
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) return 'Please enter a year';
                  final n = int.tryParse(value);
                  if (n == null || n < 2000 || n > 2100) return 'Enter valid year (2000-2100)';
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // Month Input
              TextFormField(
                controller: _monthController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Month (1-12)',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.calendar_month),
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) return 'Please enter a month';
                  final n = int.tryParse(value);
                  if (n == null || n < 1 || n > 12) return 'Enter valid month (1-12)';
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // Weekday Input
              TextFormField(
                controller: _weekdayController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Weekday (0=Mon, ... 6=Sun)',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.calendar_view_week),
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) return 'Please enter a weekday';
                  final n = int.tryParse(value);
                  if (n == null || n < 0 || n > 6) return 'Enter valid weekday (0-6)';
                  return null;
                },
              ),
              const SizedBox(height: 24),

              // Predict Button
              FilledButton.icon(
                onPressed: _isLoading ? null : _makePrediction,
                icon: _isLoading 
                    ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                    : const Icon(Icons.analytics),
                label: Text(_isLoading ? 'Processing...' : 'Predict Sales'),
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
              ),
              
              const SizedBox(height: 24),

              // Result Display
              if (_resultMessage.isNotEmpty)
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: _resultMessage.startsWith('Error') || _resultMessage.startsWith('Connection') 
                        ? Colors.red.shade50 
                        : Colors.green.shade50,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(
                      color: _resultMessage.startsWith('Error') || _resultMessage.startsWith('Connection')
                          ? Colors.red 
                          : Colors.green,
                    ),
                  ),
                  child: Text(
                    _resultMessage,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: _resultMessage.startsWith('Error') || _resultMessage.startsWith('Connection')
                          ? Colors.red.shade900 
                          : Colors.green.shade900,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
