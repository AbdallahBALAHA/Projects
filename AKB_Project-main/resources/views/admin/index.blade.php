@extends('layouts.main')
@section('MainContent')
<!-- Content Row -->
<div class="row">
    <div class="col-12">
        <table class="table table-hover">
            <thead>
            <tr class="bg-dark text-white">
                <th>Name</th>
                <th>Email</th>
                <th>Address</th>
                <th>Password</th>
                <th>Action</th>
            </tr>
            </thead>
            <tbody>
            @foreach ($customers as $customer)
                <tr>

                    <td>{{ $customer->fname }}</td>
                    <td>{{ $customer->email }}</td>
                    <td>{{ $customer->address }}</td>
                    <td>{{ $customer->password }}</td>
                    <td>
                        <a href="{{ route('customer.edit', $customer->id) }}" class="btn btn-primary btn-sm" style="background-color: green">Edit</a>
                        <form style="display: inline" action="{{ route('customer.destroy', $customer->id) }}" method="POST">
                            @csrf
                            @method('delete')
                            <button onclick="return confirm('are you sure?')" class="btn btn-danger btn-sm" style="background-color: red">Delete</button>
                        </form>
                    </td>
                </tr>
            @endforeach

            </tbody>
        </table>
        {{ $customers->links() }}
    </div>
</div>
@stop
