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
          'Year': int.parse(_yearController.text),
          'Month': int.parse(_monthController.text),
          'Weekday': int.parse(_weekdayController.text),
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _resultMessage = 'Predicted Sales: ${data['predicted_sales']} units';
        });
      } else {
        final err = jsonDecode(response.body);
        setState(() {
          _resultMessage = 'Error: ${err['detail'] ?? err['message'] ?? 'Check your inputs'}';
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
        title: const Text('Predict M01AB Sales Volume'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Card(
              elevation: 4,
              child: Padding(
                padding: const EdgeInsets.all(24.0),
                child: Form(
                  key: _formKey,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Text(
                        'Input Parameters',
                        style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 20),
                      TextFormField(
                        controller: _yearController,
                        decoration: const InputDecoration(
                          labelText: 'Year (e.g., 2026)',
                          border: OutlineInputBorder(),
                          prefixIcon: Icon(Icons.calendar_today),
                        ),
                        keyboardType: TextInputType.number,
                        validator: (value) {
                          if (value == null || value.isEmpty) return 'Enter a Year';
                          final val = int.tryParse(value);
                          if (val == null || val < 2014 || val > 2030) return 'Year must be between 2014 and 2030';
                          return null;
                        },
                      ),
                      const SizedBox(height: 16),
                      TextFormField(
                        controller: _monthController,
                        decoration: const InputDecoration(
                          labelText: 'Month (1 - 12)',
                          border: OutlineInputBorder(),
                          prefixIcon: Icon(Icons.date_range),
                        ),
                        keyboardType: TextInputType.number,
                        validator: (value) {
                          if (value == null || value.isEmpty) return 'Enter a Month';
                          final val = int.tryParse(value);
                          if (val == null || val < 1 || val > 12) return 'Month must be 1 - 12';
                          return null;
                        },
                      ),
                      const SizedBox(height: 16),
                      TextFormField(
                        controller: _weekdayController,
                        decoration: const InputDecoration(
                          labelText: 'Weekday (0=Mon - 6=Sun)',
                          border: OutlineInputBorder(),
                          prefixIcon: Icon(Icons.view_week),
                        ),
                        keyboardType: TextInputType.number,
                        validator: (value) {
                          if (value == null || value.isEmpty) return 'Enter a Weekday';
                          final val = int.tryParse(value);
                          if (val == null || val < 0 || val > 6) return 'Weekday must be 0 - 6';
                          return null;
                        },
                      ),
                      const SizedBox(height: 24),
                      SizedBox(
                        width: double.infinity,
                        height: 50,
                        child: ElevatedButton(
                          onPressed: _isLoading ? null : _makePrediction,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Theme.of(context).colorScheme.primary,
                            foregroundColor: Theme.of(context).colorScheme.onPrimary,
                          ),
                          child: _isLoading
                              ? const CircularProgressIndicator(color: Colors.white)
                              : const Text('Predict', style: TextStyle(fontSize: 18)),
                        ),
                      ),
                      const SizedBox(height: 24),
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade100,
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Colors.grey.shade300),
                        ),
                        width: double.infinity,
                        child: Text(
                          _resultMessage.isEmpty ? 'Prediction will appear here.' : _resultMessage,
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: _resultMessage.startsWith('Error') || _resultMessage.startsWith('Connection')
                                ? Colors.red
                                : Colors.green.shade800,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

