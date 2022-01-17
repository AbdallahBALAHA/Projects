@extends('layouts.main')
    @section('MainContent')
        <section class='container grey-text'>
            <h4 class='center'>Edit a User</h4>
            <form action="{{ route('customer.update', $customer->id) }}" method="POST">
                @method('put')
            <input type="hidden" name="_token" value="{{ csrf_token() }}">
                <label>Full name : </label>
                <input type="text" name="fname" value="{{$customer->fname}}">
                <label>Email : </label>
                <input type="email" name="email" value="{{$customer->email}}">
                <label>Address : </label>
                <input type="text" name="address" value="{{$customer->address}}">
                <label>Password : </label>
                <input type="text" name="password" value="{{$customer->password}}">
                <div class="center">
                    <input type="submit" name="submit" value="Edit" class="btn brand z-depth-10">
                </div>
            </form>
        </section>
@stop
