import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.dark, 
  ));
  runApp(const PharmaSalesApp());
}

class PharmaSalesApp extends StatelessWidget {
  const PharmaSalesApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Medical/Clinical Teal - Clean and professional
    const primaryColor = Color(0xFF009688); 
    const scaffoldBg = Color(0xFFF4F6F8); 

    return MaterialApp(
      title: 'PharmaCast',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: primaryColor,
          background: scaffoldBg,
          surface: Colors.white,
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: scaffoldBg,
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          surfaceTintColor: Colors.transparent,
          centerTitle: true,
          elevation: 0,
          titleTextStyle: TextStyle(
            color: Color(0xFF2D3748),
            fontSize: 20,
            fontWeight: FontWeight.w700,
            letterSpacing: -0.5,
          ),
          iconTheme: IconThemeData(color: Color(0xFF2D3748)),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide(color: Colors.grey.shade300)),
          enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide(color: Colors.grey.shade300)),
          focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: primaryColor, width: 2)),
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
          labelStyle: TextStyle(color: Colors.grey.shade600),
          prefixIconColor: primaryColor,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: primaryColor,
            foregroundColor: Colors.white,
            elevation: 0,
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            textStyle: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
          ),
        ),
        cardTheme: CardThemeData(
          color: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          margin: EdgeInsets.zero,
        ),
      ),
      home: const PredictionPage(),
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
  
  // Controllers & State
  final TextEditingController _yearController = TextEditingController();
  int? _selectedMonth;
  int? _selectedWeekday;
  
  bool _isLoading = false;
  String? _errorMessage;
  double? _predictedSales; // Main result
  double? _exactPredictedSales; // Exact result from API

  final String apiUrl = 'https://linear-regression-model-q2c6.onrender.com/predict';

  @override
  void initState() {
    super.initState();
    _yearController.text = DateTime.now().year.toString();
  }

  Future<void> _makePrediction() async {
    if (!_formKey.currentState!.validate()) return;
    
    FocusScope.of(context).unfocus();
    setState(() { 
      _isLoading = true; 
      _errorMessage = null; 
      _predictedSales = null;
      _exactPredictedSales = null;
    });

    try {
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'year': int.parse(_yearController.text),
          'month': _selectedMonth,
          'weekday': _selectedWeekday,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final val = data['predicted_sales'];
        
        double sales = 0.0;
        if (val is int) sales = val.toDouble();
        if (val is double) sales = val;

        setState(() {
          _predictedSales = sales;
          _exactPredictedSales = sales;
        });
      } else {
        setState(() => _errorMessage = 'Could not calculate prediction. Check inputs.');
      }
    } catch (e) {
      setState(() => _errorMessage = 'Failed to connect to AI server.\n(Server might be waking up...)');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.local_pharmacy_rounded, color: Color(0xFF009688)),
            SizedBox(width: 8),
            Text('PharmaCast'),
          ],
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 1. Info Card
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFFE0F2F1), // Very light teal
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: const Color(0xFFB2DFDB)),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.info_outline, size: 20, color: Color(0xFF00796B)),
                      SizedBox(width: 8),
                      Text('About M01AB Products', style: TextStyle(fontWeight: FontWeight.bold, color: Color(0xFF004D40))),
                    ],
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Anti-inflammatory and Antirheumatic non-steroids. This model forecasts demand based on seasonal patterns.',
                    style: TextStyle(color: Color(0xFF00695C), height: 1.4),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 32),
            const Text('Prediction Parameters', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Color(0xFF2D3748))),
            const SizedBox(height: 16),

            // 2. Input Form
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Form(
                  key: _formKey,
                  child: Column(
                    children: [
                      // Year Input
                      TextFormField(
                        controller: _yearController,
                        keyboardType: TextInputType.number,
                        decoration: const InputDecoration(
                          labelText: 'Year',
                          hintText: 'e.g. 2024',
                          prefixIcon: Icon(Icons.calendar_today_rounded),
                        ),
                        validator: (v) => (v == null || v.isEmpty) ? 'Required' : null,
                      ),
                      const SizedBox(height: 16),
                      
                      // Month Dropdown
                      DropdownButtonFormField<int>(
                        value: _selectedMonth,
                        decoration: const InputDecoration(
                          labelText: 'Month',
                          prefixIcon: Icon(Icons.date_range_rounded),
                        ),
                        items: const [
                          DropdownMenuItem(value: 1, child: Text("January")),
                          DropdownMenuItem(value: 2, child: Text("February")),
                          DropdownMenuItem(value: 3, child: Text("March")),
                          DropdownMenuItem(value: 4, child: Text("April")),
                          DropdownMenuItem(value: 5, child: Text("May")),
                          DropdownMenuItem(value: 6, child: Text("June")),
                          DropdownMenuItem(value: 7, child: Text("July")),
                          DropdownMenuItem(value: 8, child: Text("August")),
                          DropdownMenuItem(value: 9, child: Text("September")),
                          DropdownMenuItem(value: 10, child: Text("October")),
                          DropdownMenuItem(value: 11, child: Text("November")),
                          DropdownMenuItem(value: 12, child: Text("December")),
                        ],
                        onChanged: (v) => setState(() => _selectedMonth = v),
                        validator: (v) => v == null ? 'Required' : null,
                      ),
                      const SizedBox(height: 16),

                      // Weekday Dropdown
                      DropdownButtonFormField<int>(
                        value: _selectedWeekday,
                        decoration: const InputDecoration(
                          labelText: 'Day of Week',
                          prefixIcon: Icon(Icons.calendar_view_week_rounded),
                        ),
                        items: const [
                          DropdownMenuItem(value: 0, child: Text("Monday")),
                          DropdownMenuItem(value: 1, child: Text("Tuesday")),
                          DropdownMenuItem(value: 2, child: Text("Wednesday")),
                          DropdownMenuItem(value: 3, child: Text("Thursday")),
                          DropdownMenuItem(value: 4, child: Text("Friday")),
                          DropdownMenuItem(value: 5, child: Text("Saturday")),
                          DropdownMenuItem(value: 6, child: Text("Sunday")),
                        ],
                        onChanged: (v) => setState(() => _selectedWeekday = v),
                        validator: (v) => v == null ? 'Required' : null,
                      ),
                    ],
                  ),
                ),
              ),
            ),

            const SizedBox(height: 32),

            // 3. Action Button
            SizedBox(
              height: 54,
              child: ElevatedButton(
                onPressed: _isLoading ? null : _makePrediction,
                child: _isLoading 
                  ? const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)),
                        SizedBox(width: 12),
                        Text('Model is calculating...'),
                      ],
                    )
                  : const Text('CALCULATE FORECAST'),
              ),
            ),
            
            if (_isLoading)
               const Padding(
                 padding: EdgeInsets.only(top: 12),
                 child: Text(
                   'First run may take up to 60s if server is sleeping.',
                   textAlign: TextAlign.center,
                   style: TextStyle(color: Colors.grey, fontSize: 12),
                 ),
               ),

            const SizedBox(height: 32),

            // 4. Result Section
            if (_errorMessage != null)
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(color: Colors.red.shade50, borderRadius: BorderRadius.circular(12)),
                child: Row(
                  children: [
                    Icon(Icons.warning_amber_rounded, color: Colors.red.shade700),
                    const SizedBox(width: 12),
                    Expanded(child: Text(_errorMessage!, style: TextStyle(color: Colors.red.shade900))),
                  ],
                ),
              ),

            if (_predictedSales != null)
              Card(
                color: const Color(0xFF009688),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    children: [
                      const Text(
                        'FORECASTED VOLUME',
                        style: TextStyle(color: Colors.white70, fontSize: 12, fontWeight: FontWeight.bold, letterSpacing: 1),
                      ),
                      const SizedBox(height: 12),
                      // Rounded Answer (Main)
                      Text(
                        '${_predictedSales!.round()} Units',
                        style: const TextStyle(fontSize: 32, fontWeight: FontWeight.bold, color: Colors.white),
                      ),
                      const SizedBox(height: 8),
                      Container(height: 1, width: 60, color: Colors.white24),
                      const SizedBox(height: 8),
                      // Exact Answer (Sub)
                      Text(
                        'Exact Model Output: $_exactPredictedSales',
                        style: const TextStyle(color: Colors.white70, fontSize: 13, fontStyle: FontStyle.italic),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
