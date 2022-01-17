<?php

namespace App\Http\Controllers;

use App\Models\Customer;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;

class CustomerController extends Controller
{
    /**
     * Display a listing of the resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function index()
    {
        $value = session('fname');
        $customers = Customer::latest('id')->paginate(10);
        return view('admin.index', compact('customers','value'));
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        return view('admin.create');
    }

    /**
     * Store a newly created resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\Response
     */
    public function store(Request $request)
    {
        session([
            'fname' => $request->fname,
            'email' => $request->email,
            'address' => $request->address,
            'password' => $request->password
        ]);

        $cookie1 = cookie('fname',$request->fname, 2);
        $cookie2 = cookie('password',$request->password, 2);

        $request->validate([
            'fname' => 'required|alpha',
            'email' => 'required',
            'address' => 'required',
            'password' => 'required',
        ]);

        $customer = Customer::create([
            'fname' => $request->fname,
            'email' => $request->email,
            'address' => $request->address,
            'password' => Hash::make($request->password)
        ]);

        if($customer) {
            return redirect()->route('customer.index');
        } else  {
            return redirect()->back()->with('error', 'There is an error in your data');
        }

    }

    /**
     * Display the specified resource.
     *
     * @param  \App\Models\Customer  $customer
     * @return \Illuminate\Http\Response
     */
    public function show(Customer $customer)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     *
     * @param  \App\Models\Customer  $customer
     * @return \Illuminate\Http\Response
     */
    public function edit(Customer $customer)
    {
        $customer = Customer::find($customer->id);
        return view('admin.edit', compact('customer'));
    }

    /**
     * Update the specified resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @param  \App\Models\Customer  $customer
     * @return \Illuminate\Http\Response
     */
    public function update(Request $request, Customer $customer)
    {
        $request->validate([
            'fname' => 'required|alpha',
            'email' => 'required',
            'address' => 'required',
            'password' => 'required',
        ]);

        $customer = Customer::find($customer->id);

        $customer = $customer->update([
            'fname' => $request->fname,
            'email' => $request->email,
            'address' => $request->address,
            'password' => Hash::make($request->password),
        ]);

        if($customer) {
            return redirect()->route('customer.index');
        } else  {
            return redirect()->back()->with('error', 'There is an error in your data');
        }
    }

    /**
     * Remove the specified resource from storage.
     *
     * @param  \App\Models\Customer  $customer
     * @return \Illuminate\Http\Response
     */
    public function destroy(Customer $customer)
    {
        $customer = Customer::find($customer->id);
        $customer->delete();
        return redirect()->back();
    }
}
