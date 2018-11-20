/****************
*   JavaScript  *
*****************/
//primitive types
string
number
boolean
undefined

// for each
for (key in dic) {
  console.log(key);
  console.log(dic[key]);
}

// use map
Object.keys(dic).map(key => console.log(dic[key]));

// var is in function scope, exist variable hoist
// let is in block scope, doesn't have variable hoist
// const is in block scope, doesn't have variable hoist
console.log(m); // undefined
var m = 1;
 
console.log(n); // Uncaught ReferenceError: n is not defined
let n = 1;

console.log(v); // Uncaught ReferenceError: v is not defined
const v = 1; 
v = 2; // TypeError: Assignment to constant variable
const obj = {foo: 1};
obj.foo = 3; // valid

// evaluates expressions from left to right.
let x = 16 + 4 + "Volvo"; //20Volvo
let person = null;

// == equal value
// === equal value && equal type
null === undefined         // false
null == undefined          // true

// The only values that are not truthy in JavaScript are the following (a.k.a. falsy values):
null
undefined
0
"" (the empty string)
false
NaN

// Arrays
let cars = ["Saab", "Volvo", "BMW"];
Array(9).fill(null);
cars.slice(); // shallow copy
cars.slice(1) //["Volvo", "BMW"]
cars.slice(1, 3) //["Volvo", "BMW"]
cars.shift(); //remove "Saab"

array1.push(array2); //push into original array
array1.concat(array2); //create new array

// Object
const original = { a: 1, b: 2 };
const copy = { ...original, c: 3 }; // shallow copy => { a: 1, b: 2, c: 3 }

const { a, ...noA } = copy; // noA => { b: 2, c: 3 }

/*** SCOPE ***/
// GLOBAL

// global variables belong to the window object
// automatically global
// If you assign a value to a variable that has not been declared, it will automatically become a GLOBAL variable.
myFunction();

// code here can use carName 

function myFunction() {
    carName = "Volvo";
}

// ------

// lack of block scope: update a value between defining and invoking the function.
var v, getValue;
v = 1;
getValue = function () { return v; };
v = 2;

getValue(); // 2

// print object 
JSON.stringify(obj)


// flatten
var result = [].concat.apply([], [[1],[2,3],[4]]);
console.log(result); // [ 1, 2, 3, 4 ]

/****************
/   FUNCTION    /
*****************/

//在JavaScript中函式的設計，必定有回傳值，沒寫只是回傳undefined，相當於return undefined

// DECLARATION
function myFunction(a, b) {
    return a * b;
}

// EXPRESSION
let func = function(a, b) {
    return a * b;
};
(() => {console.log(1)})();


/*** CLOSURE ***/
// SELF INVOKING FUNCTION
// https://en.wikipedia.org/wiki/Immediately-invoked_function_expression
// perform the initialization work ONLY ONCE because after the execution we’ll loose the reference to the function.

(function () { /* ... */ })();
(function () { /* ... */ }());
(() => { /* ... */ })(); // With ES6 arrow functions (though parentheses only allowed on outside)
(function(a, b) { /* ... */ })("hello", "world");

// pros: 1) execute code once without cluttering the global namespace
//          (avoid variable hoisting from within blocks, protect against polluting the global environment and simultaneously allow public access to methods while retaining privacy for variables defined within the function. )
//
var module = (function () {
  // private
  return {
    // public
  };
})();

// functional style
const numbers = [1, 2, 3, 4, 5];
for (let num of numbers) {
  console.log(num);
}
const sum = numbers.reduce((total, num) => total + num, 0); 
const sum = numbers.reduce((total, num) => total + num, 0); // 0 is the initial accumulator total
const add1 = numbers.map(num => num + 1);
map(val => val + 1);
map((val, idx) => console.log(idx));

/****************
*  Asyncronous  *
*****************/
//XMLHttpRequest(AJAX)

/****************
*    Promise    *
*****************/
// https://zhuanlan.zhihu.com/p/30630902
// https://eyesofkids.gitbooks.io/javascript-start-es6-promise/content/contents/intro.html

// Four Fundamental Methods
Promise.resolve
Promise.reject
Promise.all
Promise.race

// new Promise( (resolve, reject) => { ... } )
const promise = new Promise((resolve, reject) => {
  // 成功時
  resolve(value)
  // 失敗時
  reject(reason)
});

//then必須回傳一個promise。promise2 = promise1.then(onFulfilled, onRejected);
promise.then(function(value) {
  // on fulfillment(已實現時)
}, function(reason) {
  // on rejection(已拒絕時)
})


// Design an executor, Revealing Constructor Pattern
function asyncFunction(value) {
    return new Promise(function(resolve, reject){
        if(value)
            resolve(value) // 已實現，成功
        else
            reject(reason) // 有錯誤，已拒絕，失敗
    });
}


// Use Promise
// 為了方便進行多個不同程式碼的連鎖，通常在只使用then方法時，都只寫第1個函式傳入參數。
// 而錯誤處理通常交給另一個catch方法來作，catch只需要一個函式傳入參數
doSomething()
.then(result => doSomethingElse(result))
.then(newResult => doThirdThing(newResult))
.then(finalResult => console.log(`Got the final result: ${finalResult}`))
.catch(failureCallback);

// Is the same as using async/await
async function foo() {
  try {
    const result = await doSomething();
    const newResult = await doSomethingElse(result);
    const finalResult = await doThirdThing(newResult);
    console.log(`Got the final result: ${finalResult}`);
  } catch(error) {
    failureCallback(error);
  }
}

// case study 1
const p1 = new Promise((resolve, reject) => {
    resolve(4)
})

p1.then((val) => {
        console.log(val) //4
        return val + 2
    })
    .then((val) => {
        console.log(val) //6
        throw new Error('error!')
    })
    .catch((err) => {      //catch無法抓到上個promise的回傳值
        console.log(err.message)
        //這裡如果有回傳值，下一個then可以抓得到
        //return 100
    })
    .then((val) => console.log(val, 'done')) //val是undefined，回傳值消息

// case study 2
// 如果在Promise中(建構函式或then中)使用其他異步回調的API時，這時候完全不能使用throw語句，Promise物件無法隱藏錯誤，導致連鎖失效
const p1 = new Promise(function(resolve, reject){
     setTimeout(function(){
         // 這裡如果用throw，是完全不會拒絕promise
         reject(new Error('error occur!'))
         //throw new Error('error occur!')
     }, 1000)
})

p1.then((val) => {
        console.log(val)
        return val + 2
    }).then((val) => console.log(val))
    .catch((err) => console.log('error:', err.message))
    .then((val) => console.log('done'))

/****************
*      ES6      *
*****************/
// Spread Operator
const fun = (x, y, z) => x + y + z;
a = [1, 2, 3]
fun(...a) //6

// Object Destructing
let node = {
        type: "Identifier",
        name: "foo"
    };

let { type, name } = node;

let a = 10, b = 5;
// assign different values using destructuring, need parenthesis
({ a, b } = node);

//A destructuring assignment expression evaluates to the right side of the expression (after the =)
function outputInfo(value) {
    console.log(value === node);        // true
}

outputInfo({ type, name } = node);

// Add new elements
let { type, name, value = true } = node;

// Use different names
let { type: localType, name: localName = 'can assign' } = node;

// Assign nested missing value
const user = {
  id: 339,
  name: 'Fred',
  age: 42
};
const {education: {school: {name = 'jack'} = {}} = {}} = user;

const [ first, , ...rest ] = [1, 2, 3, 4, 5];
// first == 1, rest == [3, 4, 5]


// Filter
var words = ['spray', 'limit', 'elite', 'exuberant', 'destruction', 'present'];
const result = words.filter(word => word.length > 6);

/****************
*     CLASS     *
*****************/

//声明式
class B {
  constructor() {}
}

//匿名表达式
let A = class {
  constructor() {}
}

//命名表达式，B可以在外部使用，而B1只能在内部使用
let B = class B1 {
  constructor() {}
}

/****************
*    ReactJS    *
*****************/

/*** CONVENTION ***/
All React components must act like pure functions with respect to their props.


const { isLoading, value, results } = this.state

// NAMING
 In React, however, it is a convention to use on[Event] names for props which represent events and handle[Event] for the methods which handle the events.

A component takes in parameters, called props (short for “properties”), and returns a hierarchy of views to display via the render method.


class ShoppingList extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      squares: Array(9).fill(null),
    };
  }

  render() {
    return (
      <div className="shopping-list">
        <h1>Shopping List for {this.props.name}</h1>
      </div>
    );
  }
}

// Example usage: 
<ShoppingList name="Mark" onClick={() => alert("react")}/>
<button onClick="() => alert("js")"></button>

// <tag /> is JSX syntax
//<div /> syntax is transformed at build time to React.createElement('div')

// FUNCTIONAL COMPONENTS
// only contain a render mehtod and don't have their own state.
function Square(props) {
  return (
    <button className="square" onClick={props.onClick}>
      {props.value}
    </button>
  );
}

/*** setState ***/
// if the state update depending on previous ones, 
// pass a function rather then obj
submit(){
  this.setState((prevState, props) => {
    return {showForm: !prevState.showForm}
  });
}

/***************************
*   Full Stack Development *
****************************/
https://stackoverflow.com/questions/51126472/how-to-organise-file-structure-of-backend-and-frontend-in-mern
