import React from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Alert } from 'react-native';

const App = () => {
    const [value, setValue] = React.useState(true); // Default to true since you check for `value`

    const handlePress = async (direction) => {
        const bodyType = { value };
        const url = `http://10.10.75.102:8000/${direction}`;  // Replace with your machine's IP address

        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
               
            });

            if (response.ok) {
                Alert.alert("Success", `${direction} command sent successfully!`);
            } else {
                Alert.alert("Error", `Failed with status: ${response.status}`);
            }
        } catch (error) {
            Alert.alert("Error", `Network request failed: ${error}`);
        }
    };

    return (
        <View style={styles.container}>
            <View style={styles.row}>
                <View style={styles.buttonContainer} />
                <TouchableOpacity
                    style={styles.button}
                    onPress={() => handlePress('front')}
                >
                    <Text style={styles.buttonText}>UP</Text>
                </TouchableOpacity>
                <View style={styles.buttonContainer} />
            </View>
            <View style={styles.row}>
                <TouchableOpacity
                    style={styles.button}
                    onPress={() => handlePress('left')}
                >
                    <Text style={styles.buttonText}>LEFT</Text>
                </TouchableOpacity>
                <View style={styles.buttonContainer} />
                <TouchableOpacity
                    style={styles.button}
                    onPress={() => handlePress('right')}
                >
                    <Text style={styles.buttonText}>RIGHT</Text>
                </TouchableOpacity>
            </View>
            <View style={styles.row}>
                <View style={styles.buttonContainer} />
                <TouchableOpacity
                    style={styles.button}
                    onPress={() => handlePress('down')}
                >
                    <Text style={styles.buttonText}>DOWN</Text>
                </TouchableOpacity>
                <View style={styles.buttonContainer} />
            </View>
            <View style={styles.row}>
                <View style={styles.buttonContainer} />
                <TouchableOpacity
                    style={styles.button}
                    onPress={() => handlePress('stop')}
                >
                    <Text style={styles.buttonText}>STOP</Text>
                </TouchableOpacity>
                <View style={styles.buttonContainer} />
            </View>
            <View style={styles.row}>
                <View style={styles.buttonContainer} />
                <TouchableOpacity
                    style={styles.button}
                    onPress={() => handlePress('capture')}
                >
                    <Text style={styles.buttonText}>CAMERA</Text>
                </TouchableOpacity>
                <View style={styles.buttonContainer} />
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#f5fcff',
    },
    row: {
        flexDirection: 'row',
    },
    buttonContainer: {
        flex: 1,
    },
    button: {
        flex: 1,
        padding: 20,
        margin: 10,
        backgroundColor: '#007AFF',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: 10,
    },
    buttonText: {
        color: '#fff',
        fontSize: 18,
    },
});

export default App;