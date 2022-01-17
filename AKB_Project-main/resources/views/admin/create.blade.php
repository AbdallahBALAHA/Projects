@extends('layouts.main')
@section('MainContent')
    <section class='container grey-text'>
        <h4 class='center'>Add a User</h4>
            <form class="white" action="{{ route('customer.store') }}" method="POST">
                <input type="hidden" name="_token" value="{{ csrf_token() }}">
            <label>Full name : </label>
            <input type="text" name="fname">
            <label>Email : </label>
            <input type="email" name="email">
            <label>Address : </label>
            <input type="text" name="address">
            <label>Password : </label>
            <input type="text" name="password">
            <div class="center">
                <input type="submit" name="submit" value="submit" class="btn brand z-depth-10">
            </div>
        </form>
    </section>
@stop
